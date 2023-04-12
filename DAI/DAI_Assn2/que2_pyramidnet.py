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
import warnings
warnings.filterwarnings("ignore")
from pyramidnet import PyramidNet, ResidualBlock

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
    
def get_model(config):
    block = ResidualBlock
    model = PyramidNet(
        num_layers = config['depth'],
        alpha = config['alpha'],
        block = block,
        num_classes = config['nclass'],
    )
    
    return model

def evaluate(model, loader, device, return_logs=False):
    correct = 0;samples =0
    model.eval()
    with torch.no_grad():
        loader_len = len(loader)
        for idx,(x,y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            scores = model(x)
            predict_prob = F.softmax(scores,dim=1)
            _,predictions = predict_prob.max(1)
            correct += (predictions == y).sum()
            samples += predictions.size(0)
        
            if return_logs:
                progress(idx+1,loader_len)
        
    acc = correct/samples
    # print(f"acc: {acc:.3f}")
    model.train()
    return acc

def train(model, train_loader, test_data, lossfunction, optimizer, n_epochs, device, return_logs=False): 
    model = model.to(device)
    tval = {'trainacc':[],"trainloss":[]}
    for epochs in range(n_epochs):
        model.train()
        cur_loss = 0
        curacc = 0
        len_train = len(train_loader)
        for idx , (data,target) in enumerate(train_loader):
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
                
        tacc = evaluate(model, test_data, device, return_logs)
      
        tval['trainacc'].append(float(curacc))
        tval['trainloss'].append(float(cur_loss))
        
        print(f"epochs: [{epochs+1}/{n_epochs}] train_acc: {curacc:.3f} train_loss: {cur_loss:.3f} test_acc: {tacc:.3f}")
        
    return model

def data(config):
    normalize_transform = torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465],[0.2470, 0.2435, 0.2616])
    if config['augment']:
        print('augmenting')
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.RandomHorizontalFlip(0.6),
            torchvision.transforms.RandomRotation(10),
            normalize_transform
        ])
    else:
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            normalize_transform
        ])
    
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        normalize_transform
    ])
    
    dataset_path = config['dataset_path']
    batch_size = config['batch_size']
    pin_memory = config['pin_memory']
    n_workers = config['num_workers']
    train_data = torchvision.datasets.CIFAR10(dataset_path, train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10(dataset_path, train=False, download=True, transform=test_transform)
    
    traindataloader = torch.utils.data.DataLoader(
        train_data,
        shuffle=True,
        batch_size = batch_size,
        pin_memory=pin_memory,
        num_workers = n_workers
    )
    
    testdataloader = torch.utils.data.DataLoader(
        test_data,
        shuffle=True,
        batch_size = batch_size,
        pin_memory=pin_memory,
        num_workers = n_workers
    )
    
    return traindataloader, testdataloader


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

        
    train_data, test_data = data(config)
     
    device = torch.device(f'cuda:{config["gpu"]}' if torch.cuda.is_available() else 'cpu')
    
    model = get_model(config)
    loss = nn.CrossEntropyLoss()
    
    if config['load']:
        from bias_utils import *
        print('==> loading pretrained model')
        print(model.load_state_dict(torch.load(config['saved_path'],map_location=device)))
        model = model.to(device)
        auc_values = evaluate_metrics(model, test_data, device, config['return_logs'])
        avg_auc = 0
        for i in auc_values.keys():
            print(f"auc for class {i}: {auc_values[i][2]:.3f}")
            avg_auc += auc_values[i][2]
        avg_auc /= len(auc_values)
        print(f'avg auc: {avg_auc:.3f}') 
            
        roc_plot(auc_values, config['roc_save_path'])
        exit(0)
    
    optimizer = optim.SGD(model.parameters(),lr=config['lr'], momentum=config['momentum'])
                            
    model = train(model, train_data, test_data, loss, optimizer, config['epochs'], device, config['return_logs'])
    
    torch.save(model.state_dict(), config['saved_path'])

    
