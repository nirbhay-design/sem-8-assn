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
    
def get_model(model_name, nclass):
    if model_name == 'resnet18':
        print(f"loading {model_name}")
        model = torchvision.models.resnet18(pretrained=True)
        model.fc = nn.Linear(in_features=512, out_features=nclass, bias=True)
    elif model_name == "shufflenet":
        print(f"loading {model_name}")
        model = torchvision.models.shufflenet_v2_x1_0(pretrained = True)
        model.fc = nn.Linear(in_features = 1024, out_features = nclass, bias=True)
    return model

def train(model, train_loader, lossfunction, optimizer, transformations, n_epochs, device, return_logs=False): 
    model = model.to(device)
    tval = {'trainacc':[],"trainloss":[]}
    model.train()
    for epochs in range(n_epochs):
        cur_loss = 0
        curacc = 0
        len_train = len(train_loader)
        for idx , (data,target) in enumerate(train_loader):
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
    train_data = torchvision.datasets.SVHN(dataset_path, split='train', download=True, transform=transform)
    test_data = torchvision.datasets.SVHN(dataset_path, split='test', download=True, transform=transform)
    
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

def evaluate(model, loader, device, transformations, return_logs=False):
    correct = 0;samples =0
    model.eval()
    with torch.no_grad():
        loader_len = len(loader)
        for idx,(x,y) in enumerate(loader):
            x = transformations(x)
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
    print(f"acc: {acc:.2f}")
    return acc
    
def evaluate_under_fgsm(model, loader, loss, device, transformations, epsilon, return_logs=False):
    correct = 0;samples =0
    model.eval()
    loader_len = len(loader)
    for idx,(x,y) in enumerate(loader):
        x = transformations(x)
        x = x.to(device)
        y = y.to(device)
        x.requires_grad = True

        normal_scores = model(x)
        cur_loss = loss(normal_scores, y)
        model.zero_grad()
        cur_loss.backward()

        adv_x = x + epsilon * x.grad.sign()
        adv_scores = model(adv_x)

        predict_prob = F.softmax(adv_scores,dim=1)
        _,predictions = predict_prob.max(1)
        correct += (predictions == y).sum()
        samples += predictions.size(0)

        if return_logs:
            progress(idx+1,loader_len)
        
    acc = correct/samples
    # print(f"after attack acc: {acc:.2f}")
    return acc

def plot_logs(logs,save_path):
    idx, values = zip(*list(logs.items()))
    values = [vl.cpu() for vl in values]
    plt.figure(figsize=(5,4))
    plt.plot(idx,values)
    plt.xlabel('epsilon')
    plt.ylabel('test accuracy')
    plt.title('acc vs epsilon')
    plt.savefig(save_path)
    
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
    
    model = get_model(config['model_name'], nclass=config['nclass'])
    transformations = StandardizeTransform()
    loss = nn.CrossEntropyLoss()
    
    
    if config['load']:
        print(model.load_state_dict(torch.load(config['model_saved_path'], map_location=device)))
        model.eval()
        model = model.to(device)
        
    else:
        optimizer = optim.SGD(model.parameters(),lr=config['lr'], momentum=config['momentum'])
                            
        model = train(model, train_data, loss, optimizer, transformations, config['epochs'], device, config['return_logs'])
        torch.save(model.state_dict(), config['model_saved_path'])

        evaluate(model, test_data, device, transformations, config['return_logs'])