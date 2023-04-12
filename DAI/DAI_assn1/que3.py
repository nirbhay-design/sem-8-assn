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

class StandardizeTransform(nn.Module):
    def __init__(self):
        super(StandardizeTransform, self).__init__()
        self.transform = None
        
    def forward(self, batch_data):
        """
        batch_data: [N, 3, W, H]
        """
        
        mean_values = []
        std_values = []
        
        mean_values.append(batch_data[:,0:1,:,:].mean())
        if batch_data.shape[1] > 1:
            mean_values.append(batch_data[:,1:2,:,:].mean())
            mean_values.append(batch_data[:,2:3,:,:].mean())
        
        std_values.append(batch_data[:,0:1,:,:].std())
        if batch_data.shape[1] > 1:
            std_values.append(batch_data[:,1:2,:,:].std())
            std_values.append(batch_data[:,2:3,:,:].std())
        
        self.transform = torchvision.transforms.Normalize(mean_values, std_values)
        
        return self.transform(batch_data)
    
class Nnet(nn.Module):
    def __init__(self, nclass=10):
        super(Nnet, self).__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(3,64,3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64,128,3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(),
            nn.Conv2d(128,64,3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))
        )
        
        self.fc = nn.Linear(64, nclass)
            
    def forward(self, x):
        x = self.layers(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x;

def train(model, orig_model, train_loader, lossfunction, optimizer, transformations, n_epochs, epsilon, device, return_logs=False): 
    model = model.to(device)
    orig_model = orig_model.to(device)
    tval = {'trainacc':[],"trainloss":[]}
    model.train()
    orig_model.eval()
    for epochs in range(n_epochs):
        cur_loss = 0
        curacc = 0
        len_train = len(train_loader)
        for idx , (data,target) in enumerate(train_loader):
            if data.shape[0] == 1:
                continue
            data = transformations(data)    
            data = data.to(device)
            target = target.to(device)
            data.requires_grad = True
            
            adv_output = orig_model(data)
            cur_loss_adv = lossfunction(adv_output, target)
            orig_model.zero_grad()
            cur_loss_adv.backward()
            
            adv_samples = data + epsilon * data.grad.sign()
            data.requires_grad = False
            
            data_shape = data.shape[0]
            data = torch.cat([data, adv_samples],dim=0)
            target = torch.cat([torch.zeros(data_shape), torch.ones(data_shape)])
            data = data.to(device)
            target = target.to(device)
            target = target.type(torch.int64)
                
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
      
        tval['trainacc'].append(float(curacc))
        tval['trainloss'].append(float(cur_loss))
        
        print(f"epochs: [{epochs+1}/{n_epochs}] train_acc: {curacc:.3f} train_loss: {cur_loss:.3f}")
        
    return model

def data(config):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.RandomHorizontalFlip(0.6),
        torchvision.transforms.RandomRotation(10),
    ])
    
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    dataset_path = config['dataset_path']
    batch_size = config['batch_size']
    pin_memory = config['pin_memory']
    n_workers = config['num_workers']
    train_data = torchvision.datasets.CIFAR10(dataset_path,train=True,download=True,transform=transform)
    test_data = torchvision.datasets.CIFAR10(dataset_path,train=False,download=True,transform=test_transform)
    
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

def evaluate(model, origmodel, loader, device, transformations, loss, epsilon, return_logs=False):
    correct = 0;samples =0
    model.eval()
    origmodel.eval()
    origmodel = origmodel.to(device)
    loader_len = len(loader)
    for idx,(x,y) in enumerate(loader):
        x = transformations(x)
        x = x.to(device)
        y = y.to(device)
        x.requires_grad = True

        normal_scores = origmodel(x)
        cur_loss = loss(normal_scores, y)
        origmodel.zero_grad()
        cur_loss.backward()

        adv_x = x + epsilon * x.grad.sign()
        
        x_shape = x.shape[0]
        x = torch.cat([x, adv_x],dim=0)
        y = torch.cat([torch.zeros(x_shape), torch.ones(x_shape)])
        x = x.to(device)
        y = y.to(device)
        y = y.type(torch.int64)

        adv_scores = model(x)

        predict_prob = F.softmax(adv_scores,dim=1)
        _,predictions = predict_prob.max(1)
        correct += (predictions == y).sum()
        samples += predictions.size(0)

        if return_logs:
            progress(idx+1,loader_len)
        
    acc = correct/samples
    print(f"acc: {acc:.2f}")
    return acc
    
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

        
    train_data, test_data = data(config)
     
    device = torch.device(f'cuda:{config["gpu"]}' if torch.cuda.is_available() else 'cpu')
    
    detmodel = torchvision.models.resnet18(pretrained=True)
    detmodel.fc = nn.Linear(in_features=512, out_features=2, bias=True)
    
    model = Nnet(config['nclass'])
    
    transformations = StandardizeTransform()
    loss = nn.CrossEntropyLoss()
    
    
    print(model.load_state_dict(torch.load(config['model_saved_path'], map_location=device)))
    model.eval()
    model = model.to(device)
    detmodel = detmodel.to(device)
        
    optimizer = optim.SGD(detmodel.parameters(),lr=config['lr'], momentum=config['momentum'])

    detmodel = train(detmodel, model, train_data, loss, optimizer, transformations, config['epochs'], config['epsilon'], device, config['return_logs'])
    # torch.save(model.state_dict(), config['det_model_saved_path'])

    evaluate(detmodel, model, test_data, device, transformations, loss, config['epsilon'], config['return_logs'])