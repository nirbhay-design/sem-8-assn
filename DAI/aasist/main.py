"""
Main script that trains, validates, and evaluates
various models including AASIST.

AASIST
Copyright (c) 2021-present NAVER Corp.
MIT license
"""
import argparse
import json
import os
import sys
import warnings
import pickle
from importlib import import_module
from pathlib import Path
from shutil import copy
from typing import Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchcontrib.optim import SWA

from data_utils import (Dataset_ASVspoof2019_train,
                        Dataset_ASVspoof2019_devNeval, genSpoof_list, CustomAudioData)
from evaluation import calculate_tDCF_EER
from utils import create_optimizer, seed_worker, set_seed, str_to_bool
import torchaudio

warnings.filterwarnings("ignore", category=FutureWarning)

def progress(current,total):
    progress_percent = (current * 50 / total)
    progress_percent_int = int(progress_percent)
    print(f" |{chr(9608)* progress_percent_int}{' '*(50-progress_percent_int)}|{current}/{total}",end='\r')
    if (current == total):
        print()

def main(args: argparse.Namespace) -> None:
    """
    Main function.
    Trains, validates, and evaluates the ASVspoof detection model.
    """
    # load experiment configurations
    with open(args.config, "r") as f_json:
        config = json.loads(f_json.read())
    model_config = config["model_config"]
    optim_config = config["optim_config"]
    optim_config["epochs"] = config["num_epochs"]
    config["freq_aug"] = "False"

    # make experiment reproducible
    set_seed(args.seed, config)

    # define database related paths
    database_path = Path(config["database_path"])

    # set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    if device == "cpu":
        raise ValueError("GPU not detected!")

    # define model architecture
    model = get_model(model_config, device)

    # define dataloaders
    trn_loader, eval_loader = get_loader(
        database_path, args.seed, config)

    # evaluates pretrained model and exit script
    if args.eval:
        model.load_state_dict(
            torch.load(config["model_path"], map_location=device))
        print("Model loaded : {}".format(config["model_path"]))
        print("Start evaluation...")
        sys.exit(0)

    # get optimizer and scheduler
    optim_config["steps_per_epoch"] = len(trn_loader)
    optimizer, scheduler = create_optimizer(model.parameters(), optim_config)
    optimizer_swa = SWA(optimizer)

    best_dev_eer = 1.
    best_eval_eer = 100.
    best_dev_tdcf = 0.05
    best_eval_tdcf = 1.
    n_swa_update = 0  # number of snapshots of model to use in SWA

    # Training
    for epoch in range(config["num_epochs"]):
        print("Start training epoch{:03d}".format(epoch))
        running_loss, running_acc = train_epoch(trn_loader, model, optimizer, device,
                                   scheduler, config)
        print(f"Loss:{running_loss:.2f} Acc:{running_acc:.2f}")
        
        optimizer_swa.update_swa()
        n_swa_update += 1
        
    acc, y_true, y_pred = eval_model(eval_loader, model, device, config)
    print(acc)
    dictt = {'ytrue':y_true, 'ypred':y_pred}
    with open(config['pickle_file'], 'wb') as f:
        pickle.dump(dictt, f)

    # torch.save(model.state_dict(),
            #    model_save_path / "swa.pth")



def get_model(model_config: Dict, device: torch.device):
    """Define DNN model architecture"""
    module = import_module("models.{}".format(model_config["architecture"]))
    _model = getattr(module, "Model")
    model = _model(model_config).to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print("no. model params:{}".format(nb_params))

    return model


def get_loader(
        database_path: str,
        seed: int,
        config: dict) -> List[torch.utils.data.DataLoader]:
    """Make PyTorch DataLoaders for train / developement / evaluation"""
    
    train_set = CustomAudioData('/DATA/sharma59/sem8assn/sem-8-assn/AudioData/audio_dataset/train')
    eval_set = CustomAudioData('/DATA/sharma59/sem8assn/sem-8-assn/AudioData/audio_dataset/test')
    trn_loader = DataLoader(train_set,
                            batch_size=config["batch_size"],
                            shuffle=True,
                            pin_memory=True
                            )
    eval_loader = DataLoader(eval_set,
                             batch_size=config["batch_size"],
                             shuffle=True,
                             pin_memory=True)

    return trn_loader, eval_loader


def train_epoch(
    trn_loader: DataLoader,
    model,
    optim: Union[torch.optim.SGD, torch.optim.Adam],
    device: torch.device,
    scheduler: torch.optim.lr_scheduler,
    config: argparse.Namespace):
    """Train the model for one epoch"""
    running_loss = 0.0
    num_total = 0.0
    running_acc = 0.0
    ii = 0
    model.train()

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    for idx, (batch_x, batch_y) in enumerate(trn_loader):
        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        _, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
        
        softmax_scores = F.softmax(batch_out, dim=1)
        _,preds = softmax_scores.max(dim=1)
        running_acc += (preds == batch_y).sum()
        
        batch_loss = criterion(batch_out, batch_y)
        running_loss += batch_loss.item() * batch_size
        optim.zero_grad()
        batch_loss.backward()
        optim.step()
        
        progress(idx+1, len(trn_loader))

    running_loss /= num_total
    running_acc /= num_total
    return running_loss, running_acc

@torch.no_grad()
def eval_model(
    trn_loader: DataLoader,
    model,
    device: torch.device,
    config: argparse.Namespace):
    """Train the model for one epoch"""
    running_acc = 0.0
    num_total = 0.0
    model.eval()
    
    y_true = None
    y_pred = None

    # set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    for batch_x, batch_y in trn_loader:
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        _, batch_out = model(batch_x, Freq_aug=str_to_bool(config["freq_aug"]))
        softmax_scores = F.softmax(batch_out, dim=1)
        _,preds = softmax_scores.max(dim=1)
        running_acc += (preds == batch_y).sum()
        
        if y_true is None:
            y_true = batch_y
            y_pred = softmax_scores
        else:
            y_true = torch.cat([y_true, batch_y],dim=0)
            y_pred = torch.cat([y_pred, softmax_scores],dim=0)

    running_acc /= num_total
    return running_acc, y_true, y_pred


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ASVspoof detection system")
    parser.add_argument("--config",
                        dest="config",
                        type=str,
                        help="configuration file",
                        required=True)
    parser.add_argument(
        "--output_dir",
        dest="output_dir",
        type=str,
        help="output directory for results",
        default="./exp_result",
    )
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="random seed (default: 1234)")
    parser.add_argument(
        "--eval",
        action="store_true",
        help="when this flag is given, evaluates given model and exit")
    parser.add_argument("--comment",
                        type=str,
                        default=None,
                        help="comment to describe the saved model")
    parser.add_argument("--eval_model_weights",
                        type=str,
                        default=None,
                        help="directory to the model weight file (can be also given in the config file)")
    main(parser.parse_args())
