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
from sklearn.metrics import classification_report,auc,roc_curve,det_curve,precision_recall_fscore_support
import torchaudio
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
    
class CustomAudioDataTrain():
    def __init__(self, data_path):
        self.data_path = data_path
        classes = {'audio_deepfakes':1, 'audio_original':0}
        self.audios = []
            
        dir_val = os.listdir(self.data_path)
        all_images = {0:[],1:[]}
        for dirr in dir_val:
            class_val = classes[dirr]
            for imgs in os.listdir(os.path.join(data_path, dirr)):
                img_path = os.path.join(data_path, dirr, imgs)
                all_images[class_val].append(img_path)
                
        self._audios(all_images[0], all_images[0], 0, 0, 1)
        self._audios(all_images[1], all_images[1], 1, 1, 1)
        self._audios(all_images[0], all_images[1], 0, 1, 0)

        self.resampled_sr = 20000
        self.target_len = 50000
        self.mels = 64
        self.fft = 1024
    
    def __len__(self):
        return len(self.audios)

    def __getitem__(self, idx):
        audio_file1, audio_file2, cls1, cls2, same = self.audios[idx]
        spectogram_audio_file1 = self._wrap(audio_file1)
        spectogram_audio_file2 = self._wrap(audio_file2)
        return spectogram_audio_file1, spectogram_audio_file2, cls1, cls2, same
    
    def _wrap(self, audio_file):
        signal, sr = torchaudio.load(audio_file)
        two_dim_signal = self._two_channel(signal)
        resampled_signal = self._resampling(self.resampled_sr, sr, two_dim_signal)
        padded_signal = self._padding(self.target_len, resampled_signal)
        spectrogram_signal = self._spectrograms(self.resampled_sr, n_mels = self.mels, n_fft=self.fft, signal=padded_signal)
        return spectrogram_signal

    def _two_channel(self, signal_vec):
        if signal_vec.shape[0] == 1:
            signal_vec = torch.cat([signal_vec, signal_vec],dim=0)
        return signal_vec

    def _resampling(self, resample_sr, oldsr, signal):
        resampler = torchaudio.transforms.Resample(oldsr, resample_sr)
        sample1 = resampler(signal[:1,:])
        sample2 = resampler(signal[1:,:])
        return torch.cat([sample1, sample2],dim=0)

    def _padding(self, pad_len, signal):
        sig_channel, signal_len = signal.shape
        if signal_len >= pad_len:
            return signal[:,:pad_len]
        else:
            remaining_padding = pad_len - signal_len
            pad_begin = remaining_padding // 2
            pad_end = remaining_padding - pad_begin
            padding_begin = torch.zeros((sig_channel, pad_begin))
            padding_end = torch.zeros((sig_channel, pad_end))
            return torch.cat([padding_begin, signal, padding_end],dim=1)

    def _spectrograms(self, sr, n_mels, n_fft, signal):
        mel_spect_transform = torchaudio.transforms.MelSpectrogram(sr, n_fft=n_fft, n_mels=n_mels)
        amp_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)
        return amp_transform(mel_spect_transform(signal))
    
    def _audios(self, img_list1, img_list2, class1, class2, same=1):
        counter = 0
        for idx, vals in enumerate(img_list1):
            for jdx, jals in enumerate(img_list2):
                if idx != jdx:
                    self.audios.append((vals, jals, class1, class2, same))
                    counter += 1
            if counter >= 3300:
                break
                    
class CustomAudioDataTest():
    def __init__(self, data_path):
        self.data_path = data_path
        classes = {'audio_deepfakes':1, 'audio_original':0}
        self.audios = []
        for cls, idx in classes.items():
            path = os.path.join(self.data_path, cls)
            audio_files = os.listdir(path)
            audio_files_cls = list(map(lambda x: (os.path.join(path,x),idx), audio_files))
            self.audios.extend(audio_files_cls)

        self.resampled_sr = 20000
        self.target_len = 50000
        self.mels = 64
        self.fft = 1024
    
    def __len__(self):
        return len(self.audios)

    def __getitem__(self, idx):
        audio_file, cls = self.audios[idx]
        signal, sr = torchaudio.load(audio_file)
        two_dim_signal = self._two_channel(signal)
        resampled_signal = self._resampling(self.resampled_sr, sr, two_dim_signal)
        padded_signal = self._padding(self.target_len, resampled_signal)
        spectrogram_signal = self._spectrograms(self.resampled_sr, n_mels = self.mels, n_fft=self.fft, signal=padded_signal)
        return spectrogram_signal, cls

    def _two_channel(self, signal_vec):
        if signal_vec.shape[0] == 1:
            signal_vec = torch.cat([signal_vec, signal_vec],dim=0)
        return signal_vec

    def _resampling(self, resample_sr, oldsr, signal):
        resampler = torchaudio.transforms.Resample(oldsr, resample_sr)
        sample1 = resampler(signal[:1,:])
        sample2 = resampler(signal[1:,:])
        return torch.cat([sample1, sample2],dim=0)

    def _padding(self, pad_len, signal):
        sig_channel, signal_len = signal.shape
        if signal_len >= pad_len:
            return signal[:,:pad_len]
        else:
            remaining_padding = pad_len - signal_len
            pad_begin = remaining_padding // 2
            pad_end = remaining_padding - pad_begin
            padding_begin = torch.zeros((sig_channel, pad_begin))
            padding_end = torch.zeros((sig_channel, pad_end))
            return torch.cat([padding_begin, signal, padding_end],dim=1)

    def _spectrograms(self, sr, n_mels, n_fft, signal):
        mel_spect_transform = torchaudio.transforms.MelSpectrogram(sr, n_fft=n_fft, n_mels=n_mels)
        amp_transform = torchaudio.transforms.AmplitudeToDB(top_db=80)
        return amp_transform(mel_spect_transform(signal))
       

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
    dataset_path = config['dataset_path']
    test_path = config['test_path']
    batch_size = config['batch_size']
    pin_memory = config['pin_memory']
    n_workers = config['num_workers']
    train_data = CustomAudioDataTrain(dataset_path)
    test_data = CustomAudioDataTest(test_path)
    
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
    y_true = None
    y_pred = None
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
            
            if y_pred is None:
                y_pred = predict_prob
                y_true = y
            else:
                y_pred = torch.cat((y_pred,predict_prob),dim=0)
                y_true = torch.cat((y_true, y), dim=0)
        
            if return_logs:
                progress(idx+1,loader_len)
        
    acc = correct/samples
    print(f"acc: {acc:.2f}")
    return acc, y_true, y_pred

def roc_det(y_true,y_pred,save_path):
    metrics = {'fpr':[],'tpr':[],'fnr':[]}
    for i in np.arange(0,1.2,0.001):
        new_y = copy.deepcopy(y_pred)
        y_pred_new = (new_y > i)
        fp = ((y_true == 0) & (y_pred_new == 1)).sum()
        tp = ((y_true == 1) & (y_pred_new == 1)).sum()
        fn = ((y_true == 1) & (y_pred_new == 0)).sum()
        tn = ((y_true == 0) & (y_pred_new == 0)).sum()
        metrics['fpr'].append(fp/(fp+tn))
        metrics['tpr'].append(tp/(tp+fn))
        metrics['fnr'].append(fn/(fn+tp))
    plt.figure(figsize=(5,4))
    # plt.plot(metrics['fpr'], metrics['tpr'], label='roc')
    plt.plot(metrics['fpr'], metrics['fnr'], label='det')
    plt.xlabel('fpr')
    plt.ylabel('tpr & fnr')
    plt.title('roc_det')
    plt.legend()
    plt.savefig(save_path)
    plt.show()
    
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
    print(len(train_data))
    print(len(test_data))
    
    device = torch.device(f'cuda:{config["gpu"]}' if torch.cuda.is_available() else 'cpu')
    
    model = torchvision.models.resnet18(pretrained=True)
    model.conv1 = nn.Conv2d(2, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    model.fc = nn.Linear(in_features=512, out_features=config['nclass'], bias=True)
    transformations = torchvision.transforms.Compose([])
    loss = Lossfunction()
    
    optimizer = optim.SGD(model.parameters(),lr=config['lr'], momentum=config['momentum'])
    
    if config['load']:
        model.load_state_dict(torch.load(config['saved_model']))
        model = model.to(device)
        acc, y_true, y_pred = evaluate(model, test_data, device, transformations, config['return_logs'])
        y_true = y_true.cpu()
        y_pred = y_pred.cpu()
        roc_det(y_true, y_pred[:,1], config['det_path'])
    else:

        model = train(model, train_data, loss, optimizer, transformations, config['epochs'], device, config['return_logs'])

        evaluate(model, test_data, device, transformations, config['return_logs'])
        torch.save(model.state_dict(),config['saved_model'])
