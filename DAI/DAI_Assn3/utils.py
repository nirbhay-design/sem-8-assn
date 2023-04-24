import yaml
from yaml.loader import SafeLoader
import torch
import torchvision
import pandas as pd
import numpy as np
import os, math
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
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings("ignore")

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
    
def progress(current,total):
    progress_percent = (current * 50 / total)
    progress_percent_int = int(progress_percent)
    print(f" |{chr(9608)* progress_percent_int}{' '*(50-progress_percent_int)}|{current}/{total}",end='\r')
    if (current == total):
        print()
    
class Fed_DATA():
    def __init__(self, data_path, n_samples, n_class, dataset, seed=42, train = True):
        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((32,32)),
            torchvision.transforms.ToTensor()
        ])
        
        self.d = dataset
        if self.d == "MNIST":
            self.data = torchvision.datasets.MNIST(
                data_path,
                train=train,
                download=True,
                transform=self.transform
            )
        elif self.d == "SVHN":
            self.data = torchvision.datasets.SVHN(
                data_path,
                split = 'train' if train else 'test',
                download=True,
                transform=self.transform
            )
        elif self.d == "CMNIST":
            self.data = torchvision.datasets.MNIST(
                data_path,
                train=train,
                download=True,
                transform=self.transform
            )
            self.c = self._generate_rgb(n_samples * n_class)
            
        self.ns = n_samples
        self.nc = n_class
        
        self.fdata = []
        self.class_cnt = {i:0 for i in range(self.nc)}
        self.flags = {i:False for i in range(self.nc)}
        for idx in range(len(self.data)):
            img, label = self.data[idx]
            if self.flags[label]:
                continue
            else:
                self.fdata.append((img, label))
                self.class_cnt[label] += 1
                if self.class_cnt[label] >= self.ns:
                    self.flags[label] = True
    
    def __len__(self):
        return len(self.fdata)
    
    def __getitem__(self, idx):
        img, label = self.fdata[idx]
        
        if self.d == 'SVHN':
            return img, label
        
        if self.d == "MNIST":
            img = torch.cat([img,img,img],dim=0)
        elif self.d == "CMNIST":
            r, g, b = self.c[idx]
            img1 = copy.deepcopy(img)
            img2 = copy.deepcopy(img)
            img3 = copy.deepcopy(img)
            img1[img1 > 0] = r
            img2[img2 > 0] = g
            img3[img3 > 0] = b
            img = torch.cat([img1,img2,img3],dim=0)
            img = img / 255
        
        return img, label
            
    def _generate_rgb(self, values):
        random.seed(42)
        c = []
        for i in range(values):
            r = random.randint(0,255)
            g = random.randint(0,255)
            b = random.randint(0,255)
            c.append((r,g,b))
        return c
    
    
def load_dataset_nonfed(config):
    
    dataset = config['dataset']
    dataset_path = config['dataset_path']
    pin_memory = config['pin_memory']
    n_workers = config['n_workers']
    img_size = config['img_size']
    batch_size = config['batch_size']
    n_trn_samples = config['trn_samples']
    n_tst_samples = config['tst_samples']
    
    data1 = Fed_DATA(
        data_path = dataset_path[0],
        n_samples= n_trn_samples, 
        n_class = config['nclass'],
        dataset = dataset[0], 
        seed=config['SEED'], 
        train = True
    )
    
    data2= Fed_DATA(
        data_path = dataset_path[1],
        n_samples= n_trn_samples, 
        n_class = config['nclass'],
        dataset = dataset[1], 
        seed=config['SEED'], 
        train = True
    )
    
    data3 = Fed_DATA(
        data_path = dataset_path[2],
        n_samples= n_trn_samples, 
        n_class = config['nclass'],
        dataset = dataset[2], 
        seed=config['SEED'], 
        train = True
    )
    
    _data1 = Fed_DATA(
        data_path = dataset_path[0],
        n_samples= n_tst_samples, 
        n_class = config['nclass'],
        dataset = dataset[0], 
        seed=config['SEED'], 
        train = False
    )
    
    _data2 = Fed_DATA(
        data_path = dataset_path[1],
        n_samples= n_tst_samples, 
        n_class = config['nclass'],
        dataset = dataset[1], 
        seed=config['SEED'], 
        train = False
    )
    
    _data3 = Fed_DATA(
        data_path = dataset_path[2],
        n_samples= n_tst_samples, 
        n_class = config['nclass'],
        dataset = dataset[2], 
        seed=config['SEED'], 
        train = False
    )
    
    train_dataset = [data1, data2, data3]
    test_dataset = [_data1, _data2, _data3]
        
    train_data = torch.utils.data.ConcatDataset(train_dataset)
    test_data = torch.utils.data.ConcatDataset(test_dataset)
    
    trn_dataloader = torch.utils.data.DataLoader(
        train_data,
        shuffle=True,
        batch_size = batch_size,
        pin_memory=pin_memory,
        num_workers = n_workers
    )
    
    tst_dataloader = torch.utils.data.DataLoader(
        test_data,
        shuffle=True,
        batch_size = batch_size,
        pin_memory=pin_memory,
        num_workers = n_workers
    )
    
    return trn_dataloader, tst_dataloader
    

def load_dataset(config):
    """
    dataset_name
    pin_memory
    n_clients
    n_workers
    batch_size
    """
    
    each_client_dataloader = []
    
    dataset = config['dataset']
    dataset_path = config['dataset_path']
    pin_memory = config['pin_memory']
    n_clients = config['n_clients']
    n_workers = config['n_workers']
    img_size = config['img_size']
    batch_size = config['batch_size']
    n_trn_samples = config['trn_samples']
    
    client1_data = Fed_DATA(
        data_path = dataset_path[0],
        n_samples= n_trn_samples, 
        n_class = config['nclass'],
        dataset = dataset[0], 
        seed=config['SEED'], 
        train = True
    )
    
    client2_data = Fed_DATA(
        data_path = dataset_path[1],
        n_samples= n_trn_samples, 
        n_class = config['nclass'],
        dataset = dataset[1], 
        seed=config['SEED'], 
        train = True
    )
    
    client3_data = Fed_DATA(
        data_path = dataset_path[2],
        n_samples= n_trn_samples, 
        n_class = config['nclass'],
        dataset = dataset[2], 
        seed=config['SEED'], 
        train = True
    )
    each_client_data = [client1_data, client2_data, client3_data]
        
    for i in range(n_clients):
        ci_dataloader = torch.utils.data.DataLoader(
            each_client_data[i],
            shuffle=True,
            batch_size = batch_size,
            pin_memory=pin_memory,
            num_workers = n_workers
        )
        each_client_dataloader.append(copy.deepcopy(ci_dataloader))
    
    return each_client_dataloader

def load_dataset_test(config):
    """
    dataset_name
    pin_memory
    n_workers
    batch_size
    img_size
    """
    
    each_client_dataloader = []
    
    dataset = config['dataset']
    dataset_path = config['dataset_path']
    pin_memory = config['pin_memory']
    n_clients = config['n_clients']
    n_workers = config['n_workers']
    img_size = config['img_size']
    batch_size = config['batch_size']
    n_tst_samples = config['tst_samples']
    
    client1_data = Fed_DATA(
        data_path = dataset_path[0],
        n_samples= n_tst_samples, 
        n_class = config['nclass'],
        dataset = dataset[0], 
        seed=config['SEED'], 
        train = False
    )
    
    client2_data = Fed_DATA(
        data_path = dataset_path[1],
        n_samples= n_tst_samples, 
        n_class = config['nclass'],
        dataset = dataset[1], 
        seed=config['SEED'], 
        train = False
    )
    
    client3_data = Fed_DATA(
        data_path = dataset_path[2],
        n_samples= n_tst_samples, 
        n_class = config['nclass'],
        dataset = dataset[2], 
        seed=config['SEED'], 
        train = False
    )
    each_client_data = [client1_data, client2_data, client3_data]
        
    for i in range(n_clients):
        ci_dataloader = torch.utils.data.DataLoader(
            each_client_data[i],
            shuffle=True,
            batch_size = batch_size,
            pin_memory=pin_memory,
            num_workers = n_workers
        )
        each_client_dataloader.append(copy.deepcopy(ci_dataloader))
    
    return each_client_dataloader

def other_log_vals(y_true, y_pred, pred_prob):
    print('---------------------------------')
    print('classification report')
    print(classification_report(y_true, y_pred))
    print('confusion matrix')
    print(confusion_matrix(y_true, y_pred))
    
    n_class = len(np.unique(y_true))

    binarize_labels = label_binarize(y_true, classes=[i for i in range(n_class)])

    auc_values = {}
    for i in range(n_class):
        fpr, tpr, _ = roc_curve(binarize_labels[:,i], pred_prob[:,i])
        auc_val = auc(fpr, tpr)
        auc_values[i] = auc_val
        print(f'auc for class {i}: {auc_val:.3f}')
    print('---------------------------------')
    

def evaluate_one_data(model, test_data, device, return_logs):
    model.eval()
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    pred_prob = []
    with torch.no_grad():
        for idx, (data, label) in enumerate(test_data):
            data, label = data.to(device), label.to(device)
            
            scores = F.softmax(model(data), dim=1)
            _,preds = scores.max(dim=1)
            correct += (preds == label).sum()
            total += data.shape[0]
            
            if return_logs:
                progress(idx+1, len(test_data))
            
            y_true.extend(label.detach().cpu().numpy())
            y_pred.extend(preds.detach().cpu().numpy())
            pred_prob.extend(scores.detach().cpu().numpy())
            
    acc = correct / total
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    pred_prob = np.array(pred_prob)
        
    return acc, y_true, y_pred, pred_prob

def evaluate_single_server(config, model, test_data, transformations, device, other_logs, return_logs):
    if isinstance(test_data, list):
        y_true = []
        y_pred = []
        pred_prob = []
        overall_acc = 0
        for data in test_data:
            acc, y_tr, y_pr, pp = evaluate_one_data(model, data, device, return_logs)
            y_true.extend(y_tr)
            y_pred.extend(y_pr)
            pred_prob.extend(pp)
            overall_acc += acc
            
        overall_acc /= len(test_data)
        
        
        if other_logs:
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            pred_prob = np.array(pred_prob)
            other_log_vals(y_true, y_pred, pred_prob)
            
        return overall_acc
            
            
    else:
        acc, y_tr, y_pr, pp = evaluate_one_data(model, test_data, device, return_logs)
        if other_logs:
            other_log_vals(y_tr, y_pr, pp)
            
        return acc

def main():
    data = Fed_DATA(
        data_path = 'datasets/MNIST',
        n_samples= 1000, 
        n_class = 10,
        dataset = "CMNIST", 
        seed=42, 
        train = True
    )
    
    print(len(data))
    
    for i in range(100):
        img, label = data[i]
        img = img.permute(1,2,0).numpy()
        print(img.max(), img.min())
        img = Image.fromarray(np.uint8(img))
        img.save(f'img_{i}.png')
        print(label)
    

    
if __name__ == "__main__":
    main()