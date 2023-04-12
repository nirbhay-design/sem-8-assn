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
import warnings
warnings.filterwarnings("ignore")

class LSTMCNN1(nn.Module):
    def __init__(self, Flstm=128, Fcnn=128, Embed_voc=10000, nclass=5, model='shufflenet'):
        super(LSTMCNN1, self).__init__()
        
        # defining LSTM layers
        self.embed_voc = Embed_voc
        self.embed_dim = 512
        self.embedding = nn.Embedding(self.embed_voc,self.embed_dim)
        self.LSTM1 = nn.LSTM(input_size=self.embed_dim, hidden_size=256, batch_first=True)
        self.d1 = nn.Dropout(p=0.5)
        self.LSTM2 = nn.LSTM(input_size=256, hidden_size=128, batch_first=True)
        self.d2 = nn.Dropout(p=0.5)
        self.LSTM3 = nn.LSTM(input_size=128, hidden_size=128, batch_first=True)
        self.FCLstm = nn.Linear(128, Flstm)
        
        # defining CNN layers
        if model == 'shufflenet':
            self.CNN = torchvision.models.shufflenet_v2_x1_0(pretrained = True)
            self.CNN.fc = nn.Linear(in_features = 1024, out_features = Fcnn, bias=True)
        elif model == "resnet":
            self.CNN = torchvision.models.resnet18(pretrained=True)
            self.CNN.fc = nn.Linear(in_features=512, out_features=Fcnn, bias=True)
            
        self.Pred = nn.Sequential(
            nn.Linear(Flstm + Fcnn, 64),
            nn.ReLU(),
            nn.Linear(64, nclass)
        )
        
    def forward(self, text, img):
        text = self.embedding(text)
        lstm_feat1, _ = self.LSTM1(text)
        lstm_feat1 = self.d1(lstm_feat1)
        lstm_feat2, _ = self.LSTM2(lstm_feat1)
        lstm_feat2 = self.d2(lstm_feat2)
        lstm_feat3, _ = self.LSTM3(lstm_feat2)
        lstm_feat = self.FCLstm(lstm_feat3[:,-1,:])
        cnn_feat = self.CNN(img) # [N, Fcnn]
        concat_feat = torch.cat([lstm_feat, cnn_feat], dim=1) # [N, Flstm + Fcnn]
        final_feat = self.Pred(concat_feat)
        
        return final_feat
    
class LSTMCNN3(nn.Module):
    def __init__(self, Flstm=128, Fcnn=128, Embed_voc=10000, nclass=5, model='shufflenet', device=None):
        super(LSTMCNN3, self).__init__()
        
        # defining LSTM layers
        self.device = device
        self.embed_voc = Embed_voc
        self.embed_dim = 512
        self.hidden_dim = 256
        self.num_layers = 3
        self.embedding = nn.Embedding(self.embed_voc,self.embed_dim)
        self.LSTM1 = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True
        )
        self.d1 = nn.Dropout(p=0.5)
        self.FCLstm = nn.Linear(self.hidden_dim, Flstm)
        
        # defining CNN layers
        if model == 'shufflenet':
            self.CNN = torchvision.models.shufflenet_v2_x1_0(pretrained = True)
            self.CNN.fc = nn.Linear(in_features = 1024, out_features = Fcnn, bias=True)
        elif model == "resnet":
            self.CNN = torchvision.models.resnet18(pretrained=True)
            self.CNN.fc = nn.Linear(in_features=512, out_features=Fcnn, bias=True)
            
        self.Pred = nn.Sequential(
            nn.Linear(Flstm + Fcnn, 64),
            nn.ReLU(),
            nn.Linear(64, nclass)
        )
        
    def forward(self, text, img):
        text = self.embedding(text)
        h1, c1 = self.init_hidden(text.shape[0])
        lstm_feat1, _ = self.LSTM1(text, (h1,c1))
        lstm_feat1 = self.d1(lstm_feat1)
        lstm_feat = self.FCLstm(lstm_feat1[:,-1,:])
        
        cnn_feat = self.CNN(img) # [N, Fcnn]
        concat_feat = torch.cat([lstm_feat, cnn_feat], dim=1) # [N, Flstm + Fcnn]
        final_feat = self.Pred(concat_feat)
        
        return final_feat
    
    def init_hidden(self, bs):
        h1 = torch.zeros(self.num_layers, bs, self.hidden_dim, device=self.device)
        c1 = torch.zeros(self.num_layers, bs, self.hidden_dim, device=self.device)
        return h1, c1
    
class LSTMCNN2(nn.Module):
    def __init__(self, Flstm=128, Fcnn=128, Embed_voc=10000, nclass=5, model='shufflenet', device= None):
        super(LSTMCNN2, self).__init__()
        
        # defining LSTM layers
        self.device= device
        self.hidden_dim = 256
        self.num_layers=3
        self.embed_voc = Embed_voc
        self.embed_dim = 512
        self.embedding = nn.Embedding(self.embed_voc,self.embed_dim)
        self.LSTM1 = nn.LSTM(
            input_size=self.embed_dim,
            hidden_size=self.hidden_dim,
            num_layers= self.num_layers,
            batch_first=True
        )
        self.d1 = nn.Dropout(p=0.5)
        self.FCLstm = nn.Linear(self.hidden_dim, Flstm)
        
        # defining CNN layers
        if model == 'shufflenet':
            self.CNN = torchvision.models.shufflenet_v2_x1_0(pretrained = True)
            self.CNN.fc = nn.Linear(in_features = 1024, out_features = Fcnn, bias=True)
        elif model == "resnet":
            self.CNN = torchvision.models.resnet18(pretrained=True)
            self.CNN.fc = nn.Linear(in_features=512, out_features=Fcnn, bias=True)
            
        self.cnn_pred = nn.Sequential(
            nn.ReLU(),
            nn.Linear(Fcnn,nclass)
        )
        
        self.lstm_pred = nn.Sequential(
            nn.ReLU(),
            nn.Linear(Flstm, nclass)
        )
            
        self.Pred = nn.Sequential(
            nn.Linear(Flstm + Fcnn, 64),
            nn.ReLU(),
            nn.Linear(64, nclass)
        )
        
    def forward(self, text, img):
        text = self.embedding(text)
        h1, c1 = self.init_hidden(text.shape[0])
        lstm_feat1, _ = self.LSTM1(text, (h1,c1))
        lstm_feat1 = self.d1(lstm_feat1)
        lstm_feat = self.FCLstm(lstm_feat1[:,-1,:])
        cnn_feat = self.CNN(img) # [N, Fcnn]
        concat_feat = torch.cat([lstm_feat, cnn_feat], dim=1) # [N, Flstm + Fcnn]
        final_feat = self.Pred(concat_feat)
        final_cnn_feat = self.cnn_pred(cnn_feat)
        final_lstm_feat = self.lstm_pred(lstm_feat)
        
        return final_feat, final_cnn_feat, final_lstm_feat
    
    def init_hidden(self, bs):
        h1 = torch.zeros(self.num_layers, bs, self.hidden_dim, device=self.device)
        c1 = torch.zeros(self.num_layers, bs, self.hidden_dim, device=self.device)
        return h1, c1

if __name__ == "__main__":
    
    model = LSTMCNN1(nclass=5)
    
    text = torch.randint(high=120, size=(32,24))
    img = torch.rand(32,3,128,100)
    
    device = torch.device('cuda:6')
    
    text = text.to(device)
    img = img.to(device)
    model = model.to(device)
    
    out = model(text, img)
    
    print(out.shape)
    
