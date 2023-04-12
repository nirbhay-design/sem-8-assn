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
import re
import string
import spacy
from LstmCNN import LSTMCNN2
from bias_utils import *
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
        
class MultiModalLoss(nn.Module):
    def __init__(self, alpha1 = 1.0, alpha2 = 1.0):
        super(MultiModalLoss, self).__init__()
        self.l1 = nn.CrossEntropyLoss()
        self.l2 = nn.CrossEntropyLoss()
        self.l3 = nn.CrossEntropyLoss()
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        
    def forward(self, feat1, feat2, feat3, target):
        # feat1 -> combine; feat2 -> cnn_feat; feat3 -> lstm_feat
        l1 = self.l1(feat1, target) 
        l2 = self.l2(feat2, target)
        l3 = self.l3(feat3, target)
        l = l1 + self.alpha1 * l2 + self.alpha2 * l3
        return l

spacy_eng = spacy.load('en_core_web_sm')

class Vocabulary():
    def __init__(self, feq):
        self.feq = feq
        self.itos = {
            0:"<PAD>",
            1:"<SOS>",
            2:"<EOS>",
            3:"<UNK>"
        }
        self.stoi = {j:i for i,j in self.itos.items()}
    
    def tokenizer(self, text):
        return [tok.text for tok in spacy_eng.tokenizer(text)]
    
    def build_voc(self, text_list):
        idx = 4
        curfeq = {}
        for text in text_list:
            for word in self.tokenizer(text):
                if word not in curfeq:
                    curfeq[word] = 1
                else:
                    curfeq[word] += 1
                if curfeq[word] == self.feq:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1
    
    def numeric(self, text):
        tokenize_text = self.tokenizer(text)
        
        numeric_val = [self.stoi['<SOS>']]
        numeric_val += [self.stoi[token] if token in self.stoi else self.stoi['<UNK>'] for token in tokenize_text]
        numeric_val += [self.stoi['<EOS>']]
        
        return numeric_val

class CrisisMMD():
    def __init__(self, base_path, path, freq, transforms=None, vocab=None):
        self.path = base_path
        self.data = pd.read_table(os.path.join(base_path, path))
        self.classes =  {
            'california_wildfires':0,
            'hurricane_harvey':1,
            'iraq_iran_earthquake':2,
            'mexico_earthquake':3,
            'srilanka_floods':4
        }
        self.data = self.data.loc[self.data['event_name'].isin(list(self.classes.keys())),:]
        self.data = self.data.loc[:,['event_name','tweet_text','image']]
        self.data['event_name'] = self.data['event_name'].apply(lambda x:self.classes[x])
        self.data['tweet_text'] = self.data['tweet_text'].apply(lambda x: self.preprocess(x))
        
        self.tt = self.data['tweet_text'].tolist()
        
        if vocab is None:
            self.voc = Vocabulary(freq)
            self.voc.build_voc(self.tt)
        else:
            self.voc = vocab
        
        self.data = np.array(self.data)
        
        self.transform = transforms
        if self.transform is None:
            self.transform = torchvision.transforms.ToTensor()
        
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        label, tweet, image = self.data[idx,:]
        tweet_to_num = torch.tensor(self.voc.numeric(tweet))
        img = Image.open(os.path.join(self.path, image)).convert("RGB")
        img = self.transform(img)
        
        return (tweet_to_num, img, label)
    
    def preprocess(self, text):
        text = text.lower()#converting string to lowercase
        res1 = re.sub(r'((http|https)://|www.).+?(\s|$)',' ',text)#removing links
        res2 = re.sub(f'[{string.punctuation}]+',' ',res1)#removing non english and special characters
        res3 = re.sub(r'[^a-z0-9A-Z\s]+',' ',res2)#removing anyother that is not consider in above
        res4 = re.sub(r'(\n)+',' ',res3)#removing all new line characters
        res = re.sub(r'\s{2,}',' ',res4)#remove all the one or more consecutive occurance of sapce
        res = res.strip()
        return res
    
class CustomCollate():
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx
    
    def __call__(self, batch):
        
        imgs = []
        text = []
        labels = []
        
        for bt in batch:
            imgs.append(bt[1].unsqueeze(0))
            text.append(bt[0])
            labels.append(bt[2])
            
        imgs = torch.cat(imgs, dim=0)
        padded_text = torch.nn.utils.rnn.pad_sequence(text, batch_first = True, padding_value = self.pad_idx)
        labels = torch.tensor(labels)
        return padded_text, imgs, labels
    
def Dataset(config):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(config['img_size']),
        torchvision.transforms.ToTensor()
    ])
    
    train_data = CrisisMMD(
        config['base_path'],
        config['train_data_path'],
        config['freq'],
        transforms=transform
    )
    
    test_data = CrisisMMD(
        config['base_path'],
        config['test_data_path'],
        config['freq'],
        transforms=transform,
        vocab=train_data.voc
    )
    
    pad_value = train_data.voc.stoi['<PAD>']
    vocab = len(train_data.voc.stoi)
    
    traindataloader = torch.utils.data.DataLoader(
        train_data,
        shuffle=True,
        batch_size=config['batch_size'],
        pin_memory=True,
        num_workers = config['num_workers'],
        collate_fn = CustomCollate(pad_value)
    )
    
    testdataloader = torch.utils.data.DataLoader(
        test_data,
        shuffle=True,
        batch_size=config['batch_size'],
        pin_memory=config['pin_memory'],
        num_workers = config['num_workers'],
        collate_fn = CustomCollate(pad_value)
    )
    
    return traindataloader, testdataloader, vocab

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
        for idx,(text, img, y) in enumerate(loader):
            text = text.to(device)
            img = img.to(device)
            y = y.to(device)

            scores, scores_cnn, scores_lstm = model(text, img)
            avg_logits = (scores + scores_cnn + scores_lstm)/3
            
            curacc1 += cur_acc(scores, y)/loader_len
            curacc2 += cur_acc(scores_cnn, y)/loader_len
            curacc3 += cur_acc(scores_lstm, y)/loader_len
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
        for idx , (text, img ,target) in enumerate(train_loader):
            text = text.to(device)
            img = img.to(device)
            target = target.to(device)
                
            scores, scores_cnn, scores_lstm = model(text, img)    
            loss = lossfunction(scores, scores_cnn, scores_lstm, target)             

            cur_loss += loss.item() / (len_train)
            curacc1 += cur_acc(scores, target)/len_train
            curacc2 += cur_acc(scores_cnn, target)/len_train
            curacc3 += cur_acc(scores_lstm, target)/len_train
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()
        
            if return_logs:
                progress(idx+1,len(train_loader))
                
        tacc1, tacc2, tacc3, ensacc = evaluate(model, test_data, device, return_logs)
      
        print(f"epochs: [{epochs+1}/{n_epochs}] FusionTrn: {curacc1:.3f} CNNTrn: {curacc2:.3f} LstmTrn: {curacc3:.3f} train_loss: {cur_loss:.3f} FusionTst: {tacc1:.3f} CNNTst: {tacc2:.3f} LstmTst: {tacc3:.3f} ENS_tst: {ensacc:.3f}")
        
    return model
        
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
    
    train_data, test_data, vocab = Dataset(config)
    
    device = torch.device(f'cuda:{config["gpu"]}' if torch.cuda.is_available() else 'cpu')
    
    print(len(train_data))
    print(len(test_data))
    print(vocab)
    
    model = LSTMCNN2(
        Flstm=128,
        Fcnn=128,
        Embed_voc=vocab,
        nclass=config['nclass'],
        model=config['cnn_type'],
        device = device
    )
    
    if config['load']:
        print('==> loading pretrained model')
        print(model.load_state_dict(torch.load(config['saved_path'],map_location=device)))
        model = model.to(device)
        auc_values = evaluate_mtl_metrics(model, test_data, device, config['return_logs'])
        avg_auc = 0
        for i in auc_values.keys():
            print(f"auc for class {i}: {auc_values[i][2]:.3f}")
            avg_auc += auc_values[i][2]
        avg_auc /= len(auc_values)
        print(f'avg auc: {avg_auc:.3f}')  
        roc_plot(auc_values, config['roc_save_path'])
        exit(0)
        
    
    loss = MultiModalLoss(alpha1 = config['alpha1'], alpha2 = config['alpha2'])
    
    if config['opt'] == "SGD":
        print('using sgd')
        optimizer = optim.SGD(model.parameters(),lr=config['lr'], momentum=config['momentum'])
    else:
        print('using adam')
        optimizer = optim.Adam(model.parameters(), lr = config['lr'])
                            
    model = train(model, train_data, test_data, loss, optimizer, config['epochs'], device, config['return_logs'])
    
    torch.save(model.state_dict(), config['saved_path'])

    
    
    
    





