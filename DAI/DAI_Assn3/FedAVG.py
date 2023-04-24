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
from collections import OrderedDict
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report,auc,roc_curve,precision_recall_fscore_support
from utils import evaluate_single_server
from utils import load_dataset, load_dataset_test
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

def aggregate_models(server_model, clients_model):
    update_state = OrderedDict()
    n_clients = len(clients_model)
    for k, client in enumerate(clients_model):
        local_state = client.state_dict()
        for key in server_model.state_dict().keys():
            if k == 0:
                update_state[key] = local_state[key] / n_clients 
            else:
                update_state[key] += local_state[key] / n_clients

    print(server_model.load_state_dict(update_state))
    
    return server_model

def return_model(model_name:str, nclass:int):
    if model_name == 'Resnet18':
        global_model = torchvision.models.resnet18(pretrained=True)
        global_model.fc = nn.Linear(in_features=512, out_features=nclass, bias=True)
    return global_model

def return_essentials(config, client_data, client_test_data):
    n_clients = config['n_clients']
    lr = config['lr']
    nclass = config['nclass']
    model_name = config['client_model']
    
    client_essential_arr = []
    for i in range(n_clients):
        if config['return_logs']:
            progress(i+1,n_clients)
            
        lossfunction = nn.CrossEntropyLoss()
        
        client_model = return_model(model_name, nclass)
        
        optimizer = optim.SGD(params=client_model.parameters(),lr=lr,momentum=0.9)
        essential = {
            "lossfun": lossfunction,
            "optimizer": optimizer,
            "model":client_model,
            "data":client_data[i],
            "tdata": client_test_data[i]
        }
        client_essential_arr.append(essential)
        
    return client_essential_arr
    
def train_client(model, train_loader, lossfunction, optimizer, transformations, n_epochs, device, return_logs=False): 
    model = model.to(device)
    tval = {'trainacc':[],"trainloss":[]}
    model.train()
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

            scores = model(data)    
            loss = lossfunction(scores,target)            

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
    return model, tval

def train_server(model, local_models, device): 
    
    model = aggregate_models(model, local_models)
    
    return model

def fedavg(config, client_essential, device, server_device, transformations, test_data, return_logs=True):
    
    total_iterations = config['total_iterations']
    client_iterations = config['client_iterations']
    n_clients = config['n_clients']
    nclass = config['nclass']
    img_size = config['img_size']
    other_logs = False
    
    start_time = time.perf_counter()
    
    acc_values = {'server':[],'client1':[],'client2':[],'client3':[]}
    server_model = return_model(config['model'], nclass)
    server_model = server_model.to(server_device)
    
    for idx in range(total_iterations):
        print(f"iteration [{idx+1}/{total_iterations}]")
        
        for jdx in range(n_clients):
            print(f"############## client {jdx} ##############")
            client_essential[jdx]['optimizer'] = optim.SGD(params=client_essential[jdx]['model'].parameters(),lr=config['lr'],momentum=0.9)
            
            client_model, log = train_client(
                client_essential[jdx]['model'],
                client_essential[jdx]['data'],
                client_essential[jdx]['lossfun'],
                client_essential[jdx]['optimizer'],
                transformations,
                n_epochs = client_iterations,
                device=device,
                return_logs=return_logs)
            client_essential[jdx]['model'] = copy.deepcopy(client_model)
            
        print("############## server ##############")
        cur_server_model = train_server(
            server_model,
            [client_essential[i]['model'] for i in range(n_clients)],
            server_device
        )
        
        server_model = copy.deepcopy(cur_server_model)
        
        if idx == total_iterations-1:
            other_logs = True
        
        server_acc = evaluate_single_server(
            config,
            server_model,
            test_data,
            transformations,
            server_device,
            other_logs = other_logs,
            return_logs = config['return_logs']
        )
        
        c1_acc = evaluate_single_server(
            config,
            client_essential[0]['model'],
            client_essential[0]['tdata'],
            transformations,
            server_device,
            other_logs = other_logs,
            return_logs = config['return_logs']
        )
        
        c2_acc = evaluate_single_server(
            config,
            client_essential[1]['model'],
            client_essential[1]['tdata'],
            transformations,
            server_device,
            other_logs = other_logs,
            return_logs = config['return_logs']
        )
        
        c3_acc = evaluate_single_server(
            config,
            client_essential[2]['model'],
            client_essential[2]['tdata'],
            transformations,
            server_device,
            other_logs = other_logs,
            return_logs = config['return_logs']
        )
        
        print(f"S, C1, C2, C3 Acc: {server_acc:.3f}, {c1_acc:.3f}, {c2_acc:.3f}, {c3_acc:.3f}")
        
        acc_values['server'].append(server_acc)
        acc_values['client1'].append(c1_acc)
        acc_values['client2'].append(c2_acc)
        acc_values['client3'].append(c3_acc)
        
        for kdx in range(n_clients):
            client_essential[kdx]['model'] = copy.deepcopy(server_model)
            client_essential[kdx]['model'].train()
                
    end_time = time.perf_counter()
    elapsed_time = int(end_time - start_time)
    hr = elapsed_time // 3600
    mi = (elapsed_time - hr * 3600) // 60
    print(f"training done in {hr} H {mi} M")
    return server_model, acc_values

def plot_logs(dict_logs, save_path, iters):
    plt.figure(figsize=(5,4))
    for key in dict_logs.keys():
        value = [i.cpu() for i in dict_logs[key]]
        plt.plot(list(range(len(value))), value, label=key)
    plt.xlabel('# of Rounds')
    plt.ylabel('Test Acc')
    plt.title('Acc vs Rounds')
    plt.legend()
    plt.xlim(0,iters-1)
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

    client_data = load_dataset(config)
    test_data = load_dataset_test(config)
     
    client_essential = return_essentials(config, client_data, test_data)
    device = torch.device(f'cuda:{config["gpu"]}' if torch.cuda.is_available() else 'cpu')
    server_device = torch.device(f'cuda:{config["server_gpu"]}' if torch.cuda.is_available() else 'cpu')
    transformations = transforms.Compose([])
    global_model, logs = fedavg(
        config,
        client_essential,
        device,
        server_device,
        transformations,
        test_data,
        return_logs=config["return_logs"])
    
    
    plot_logs(logs, config['acc_path'], config['total_iterations'])
    with open(config['acc_pkl'] , 'wb') as f:
        pickle.dump(logs, f)
    
    
    
