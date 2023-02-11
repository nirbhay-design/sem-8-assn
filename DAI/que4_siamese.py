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
    
class Lossfunction(nn.Module):
    def __init__(self):
        super(Lossfunction, self).__init__()
        self.ce = nn.CrossEntropyLoss()
    def forward(self, x, y, lx, ly, z):
        norm = torch.norm(x-y,dim=1)
        l_siamese = torch.mean(z * norm - (1-z) * norm) 
        l_ce_x = self.ce(x,lx)
        l_ce_y = self.ce(y,ly)
        return l_siamese + l_ce_x + l_ce_y
    
class TrainDataset():
    def __init__(self, data_path, transformations):
        """
        data path includes path/df, path/oimg
        """
        class_map = {"df": 1, "oimg":0}
        dir_val = os.listdir(data_path)
        all_images = {0:[],1:[]}
        for dirr in dir_val:
            class_val = class_map[dirr]
            for imgs in os.listdir(os.path.join(data_path, dirr)):
                img_path = os.path.join(data_path, dirr, imgs)
                all_images[class_val].append(img_path)
        
        self.images = []
                
        self._getimgs(all_images[0], all_images[0], 0, 0, 1)
        self._getimgs(all_images[1], all_images[1], 1, 1, 1)
        self._getimgs(all_images[0], all_images[1], 0, 1, 0)
        
        self.transformations = transformations
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img1, img2, lbl1, lbl2, same = self.images[idx]
        img1 = Image.open(img1).convert("RGB")
        img2 = Image.open(img2).convert("RGB")
        img1_tensor = self.transformations(img1)
        img2_tensor = self.transformations(img2)
        return img1_tensor, img2_tensor, lbl1, lbl2, same
    
    def _getimgs(self, img_list1, img_list2, class1, class2, same=1):
        counter = 0
        for idx, vals in enumerate(img_list1):
            for jdx, jals in enumerate(img_list2):
                if idx != jdx:
                    self.images.append((vals, jals, class1, class2, same))
                    counter += 1
            if counter >= 3300:
                break
                    
class TestDataset():
    def __init__(self, data_path, transformations):
        """
        data path includes path/df, path/oimg
        """
        class_map = {"df": 1, "oimg":0}
        dir_val = os.listdir(data_path)
        self.images = []
        for dirr in dir_val:
            class_val = class_map[dirr]
            for imgs in os.listdir(os.path.join(data_path, dirr)):
                img_path = os.path.join(data_path, dirr, imgs)
                self.images.append((img_path,class_val))
        
        self.transformations = transformations
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img1, lbl1 = self.images[idx]
        img1 = Image.open(img1).convert("RGB")
        img1_tensor = self.transformations(img1)
        return img1_tensor, lbl1

def train(model, train_loader, lossfunction, optimizer, transformations, n_epochs, device, return_logs=False): 
    model = model.to(device)
    tval = {'trainacc1':[],"trainloss":[], "trainacc2":[]}
    model.train()
    for epochs in range(n_epochs):
        cur_loss = 0
        curacc1 = 0
        curacc2 = 0
        len_train = len(train_loader)
        for idx , (data1, data2, target1, target2, same) in enumerate(train_loader):
            data1 = transformations(data1)    
            data1 = data1.to(device)
            target1 = target1.to(device)
            
            data2 = transformations(data2)    
            data2 = data2.to(device)
            target2 = target2.to(device)
            
            same = same.to(device)
        
            scores1 = model(data1)    
            scores2 = model(data2)
            
            loss = lossfunction(scores1, scores2, target1, target2, same)   

            cur_loss += loss.item() / (len_train)
            
            scores1 = F.softmax(scores1,dim = 1)
            _,predicted = torch.max(scores1,dim = 1)
            correct = (predicted == target1).sum()
            samples = scores1.shape[0]
            curacc1 += correct / (samples * len_train)
            
            scores2 = F.softmax(scores2,dim = 1)
            _,predicted = torch.max(scores2,dim = 1)
            correct = (predicted == target2).sum()
            samples = scores2.shape[0]
            curacc2 += correct / (samples * len_train)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
        
            if return_logs:
                progress(idx+1,len(train_loader))
      
        tval['trainacc1'].append(float(curacc1))
        tval['trainacc2'].append(float(curacc2))
        tval['trainloss'].append(float(cur_loss))
        
        print(f"epochs: [{epochs+1}/{n_epochs}] train_acc1: {curacc1:.3f} train_acc2: {curacc2:.3f} train_loss: {cur_loss:.3f}")
        
    return model

def data(config):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(config['img_size']),
        # torchvision.transforms.RandomHorizontalFlip(0.6),
        torchvision.transforms.RandomRotation(10),
        torchvision.transforms.ToTensor()   
    ])
    
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(config['img_size']),
        torchvision.transforms.ToTensor()
    ])
    dataset_path = config['dataset_path']
    test_path = config['test_path']
    batch_size = config['batch_size']
    pin_memory = config['pin_memory']
    n_workers = config['num_workers']
    train_data = TrainDataset(dataset_path, transform)
    test_data = TestDataset(test_path, test_transform)
    
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
    
    model = torchvision.models.resnet18(pretrained=True)
    model.fc = nn.Linear(in_features=512, out_features=config['nclass'], bias=True)
    transformations = StandardizeTransform()
    loss = Lossfunction()
    
    optimizer = optim.SGD(model.parameters(),lr=config['lr'], momentum=config['momentum'])

    model = train(model, train_data, loss, optimizer, transformations, config['epochs'], device, config['return_logs'])

    evaluate(model, test_data, device, transformations, config['return_logs'])