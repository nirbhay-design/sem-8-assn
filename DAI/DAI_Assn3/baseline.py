import yaml
from yaml.loader import SafeLoader
import torch
import torchvision
import pandas as pd
import numpy as np
import os
from PIL import Image, ImageDraw
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pickle
import random
import time,json
import copy,sys
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report,auc,roc_curve,precision_recall_fscore_support
from utils import load_dataset_nonfed, evaluate_single_server, StandardizeTransform
import warnings
warnings.filterwarnings("ignore")

def yaml_loader(yaml_file):
    with open(yaml_file,'r') as f:
        config_data = yaml.load(f,Loader=SafeLoader)
        
    return config_data

def progress(current,total):
    progress_percent = (current * 50 / total)
    progress_percent_int = int(progress_percent)
    print(f" |{chr(9608)* progress_percent_int}{' '*(50-progress_percent_int)}|{current}/{total}",end='\r')
    if (current == total):
        print()

def return_model(model_name:str, nclass:int):
    if model_name == 'Resnet18':
        global_model = torchvision.models.resnet18(pretrained=True)
        global_model.fc = nn.Linear(in_features=512, out_features=nclass, bias=True)
    return global_model


def train(model, train_loader, test_loader, lossfunction, optimizer, transformations, n_epochs, device, return_logs=False): 
    model = model.to(device)
    tval = {'trainacc':[],"trainloss":[], 'test_acc':[]}
    
    for epochs in range(n_epochs):
        model.train()
        cur_loss = 0
        curacc = 0
        len_train = len(train_loader)
        for idx , (data,target) in enumerate(train_loader):
            if data.shape[0] == 1:
                continue
            data = transformations(data)    
            data = data.to(device)
            target = target.to(device)
                
            scores = model(data)    
            loss = lossfunction(scores, target)            

            cur_loss += loss.item() / (len_train)
            scores = F.softmax(scores,dim = 1)
            _,predicted = torch.max(scores,dim = 1)
            correct = (predicted == target).sum()
            samples = scores.shape[0]
            curacc += correct / (samples * len_train)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if return_logs:
                progress(idx+1,len(train_loader))
                
        cur_test_acc = evaluate_single_server(
            config,
            model,
            test_loader,
            transformations,
            device,
            other_logs=False,
            return_logs=return_logs
        )
        
        tval['test_acc'].append(float(cur_test_acc))
        tval['trainacc'].append(float(curacc))
        tval['trainloss'].append(float(cur_loss))
        
        print(f"epochs: [{epochs+1}/{n_epochs}] train_acc: {curacc:.3f} train_loss: {cur_loss:.3f} test_acc: {cur_test_acc:.3f}")
        
    return model, tval


def plot_logs(logs,save_path):
    values = logs['test_acc']
    plt.figure(figsize=(5,4))
    plt.plot(list(range(len(values))), values)
    plt.xlabel('Epochs')
    plt.ylabel('Test Acc')
    plt.title('Acc vs Epochs')
    plt.xlim(0, len(values)-1)
    plt.savefig(save_path, format='svg')
    
if __name__ == "__main__":
    
    config = yaml_loader(sys.argv[1])
    
    random.seed(config["SEED"])
    np.random.seed(config["SEED"])
    torch.manual_seed(config["SEED"])
    torch.cuda.manual_seed(config["SEED"])
    torch.backends.cudnn.benchmarks = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print("environment: ")
    print(f"YAML: {sys.argv[1]}")
    for key, value in config.items():
        print(f"==> {key}: {value}")
        
    train_data, test_data = load_dataset_nonfed(config)
     
    device = torch.device(f'cuda:{config["gpu"]}' if torch.cuda.is_available() else 'cpu')
    
    model = return_model(config['model'], config['nclass'])
    transformations = StandardizeTransform()
    loss = nn.CrossEntropyLoss()
    
    optimizer = optim.SGD(model.parameters(),lr=config['lr'], momentum=config['momentum'])

    model, logs = train(model, train_data, test_data, loss, optimizer, transformations, config['epochs'], device, config['return_logs'])
    
    cur_test_acc = evaluate_single_server(
            config,
            model,
            test_data,
            transformations,
            device,
            other_logs=True,
            return_logs=config['return_logs']
        )
    plot_logs(logs, config['save_path'])

