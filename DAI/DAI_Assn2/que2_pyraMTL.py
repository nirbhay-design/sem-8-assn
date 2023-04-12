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
        
class MTLLoss(nn.Module):
    def __init__(self, nclass=10):
        super(MTLLoss, self).__init__()
        self.l1 = nn.CrossEntropyLoss()
        self.l2 = nn.KLDivLoss()
        self.l3 = nn.MSELoss()
        self.nclass = nclass
        
    def forward(self, x1, x2, x3, t):
        # applying kldiv in x2, ce on x1, and l2 on x3
        lsmax_x2 = F.log_softmax(x2,dim=1)
        smax_x3 = F.softmax(x3, dim=1)
        oh_t = self.cls_to_prob(t)
        
        return self.l1(x1, t) + self.l2(lsmax_x2, oh_t) + self.l3(smax_x3, oh_t)
        
    def cls_to_prob(self, x):
        xsh = x.shape[0]
        idx_list = list(range(xsh))
        new_x = torch.zeros(xsh, self.nclass,device=x.device)
        new_x[idx_list, x] = 1
        return new_x
        
class MTL1(nn.Module):
    def __init__(self, in_features=128, nclass=10):
        super(MTL1,self).__init__()
        self.in_features = in_features
        self.nclass = nclass
        
        self.mtl = nn.Sequential(
            nn.Linear(self.in_features, 64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32,self.nclass)
        )
        
    def forward(self,x):
        return self.mtl(x)
    
class MTL2(nn.Module):
    def __init__(self, in_features=128, nclass=10):
        super(MTL2,self).__init__()
        self.in_features = in_features
        self.nclass = nclass
        
        self.mtl = nn.Sequential(
            nn.Linear(self.in_features, 64),
            nn.ReLU(),
            nn.Linear(64,self.nclass)
        )
        
    def forward(self,x):
        return self.mtl(x)    
        
class MTL3(nn.Module):
    def __init__(self, in_features=128, nclass=10):
        super(MTL3,self).__init__()
        self.in_features = in_features
        self.nclass = nclass
        
        self.mtl = nn.Sequential(
            nn.Linear(self.in_features, 32),
            nn.ReLU(),
            nn.Linear(32,self.nclass)
        )
        
    def forward(self,x):
        return self.mtl(x) 
    
class PyramidNetMTL(nn.Module):
    def __init__(self, config):
        super(PyramidNetMTL, self).__init__()
        self.block = ResidualBlock
        self.model = PyramidNet(
            num_layers = config['depth'],
            alpha = config['alpha'],
            block = self.block,
            num_classes = config['nclass'],
        )
        self.model.fc_out = nn.Linear(in_features=62, out_features=128, bias=True)
        
        self.mtl1 = MTL1(nclass=config['nclass'])
        self.mtl2 = MTL2(nclass=config['nclass'])
        self.mtl3 = MTL3(nclass=config['nclass'])
        
    def forward(self, x):
        x = self.model(x)
        x1 = self.mtl1(x)
        x2 = self.mtl2(x)
        x3 = self.mtl3(x)
        
        return x1, x2, x3
    
def cur_acc(scores, target):
    scores = F.softmax(scores,dim = 1)
    _,predicted = torch.max(scores,dim = 1)
    correct = (predicted == target).sum()
    samples = scores.shape[0]
    
    return correct / samples
        
def evaluate(model, loader, device, return_logs=False):
    
    curacc1 = 0; curacc2=0;curacc3=0;curaccens=0
    model.eval()
    with torch.no_grad():
        loader_len = len(loader)
        for idx,(x,y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            scores1, scores2, scores3 = model(x)
            avg_logits = (scores1 + scores2 + scores3)/3
            
            curacc1 += cur_acc(scores1, y)/loader_len
            curacc2 += cur_acc(scores2, y)/loader_len
            curacc3 += cur_acc(scores3, y)/loader_len
            curaccens += cur_acc(avg_logits, y)/loader_len
            
            if return_logs:
                progress(idx+1,loader_len)
        
    model.train()
    return curacc1, curacc2, curacc3, curaccens

def train(model, train_loader, test_data, lossfunction, optimizer, n_epochs, device, return_logs=False): 
    model = model.to(device)
    for epochs in range(n_epochs):
        model.train()
        cur_loss = 0
        curacc1 = 0
        curacc2 = 0
        curacc3 = 0
        len_train = len(train_loader)
        for idx , (data,target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
                
            scores1, scores2, scores3 = model(data)    
            loss = lossfunction(scores1, scores2, scores3, target)            

            cur_loss += loss.item() / (len_train)
            curacc1 += cur_acc(scores1, target)/len_train
            curacc2 += cur_acc(scores2, target)/len_train
            curacc3 += cur_acc(scores3, target)/len_train
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            if return_logs:
                progress(idx+1,len(train_loader))
                
        tacc1, tacc2, tacc3, ensacc = evaluate(model, test_data, device, return_logs)
      
        print(f"epochs: [{epochs+1}/{n_epochs}] MTL1_Trn: {curacc1:.3f} MTL2_Trn: {curacc2:.3f} MTL3_Trn: {curacc3:.3f} train_loss: {cur_loss:.3f} MTL1_tst: {tacc1:.3f} MTL2_tst: {tacc2:.3f} MTL3_tst: {tacc3:.3f} ENS_tst: {ensacc:.3f}")
        
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
    
    model = PyramidNetMTL(config)
    
    loss = MTLLoss(config['nclass'])
    
    if config['load']:
        from bias_utils import *
        print('==> loading pretrained model')
        print(model.load_state_dict(torch.load(config['saved_path'],map_location=device)))
        model = model.to(device)
        auc_values = evaluate_mtl_metrics_pyr(model, test_data, device, config['return_logs'])
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

    
