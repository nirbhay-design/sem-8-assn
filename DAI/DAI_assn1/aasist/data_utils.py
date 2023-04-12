import numpy as np
import soundfile as sf
import torch
from torch import Tensor
from torch.utils.data import Dataset
import torchaudio
import os

___author__ = "Hemlata Tak, Jee-weon Jung"
__email__ = "tak@eurecom.fr, jeeweon.jung@navercorp.com"


def genSpoof_list(dir_meta, is_train=False, is_eval=False):

    d_meta = {}
    file_list = []
    with open(dir_meta, "r") as f:
        l_meta = f.readlines()

    if is_train:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list

    elif is_eval:
        for line in l_meta:
            _, key, _, _, _ = line.strip().split(" ")
            #key = line.strip()
            file_list.append(key)
        return file_list
    else:
        for line in l_meta:
            _, key, _, _, label = line.strip().split(" ")
            file_list.append(key)
            d_meta[key] = 1 if label == "bonafide" else 0
        return d_meta, file_list


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x


def pad_random(x: np.ndarray, max_len: int = 64600):
    x_len = x.shape[0]
    # if duration is already long enough
    if x_len >= max_len:
        stt = np.random.randint(x_len - max_len)
        return x[stt:stt + max_len]

    # if too short
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (num_repeats))[:max_len]
    return padded_x


class Dataset_ASVspoof2019_train(Dataset):
    def __init__(self, list_IDs, labels, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
           self.labels      : dictionary (key: utt key, value: label integer)"""
        self.list_IDs = list_IDs
        self.labels = labels
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        X_pad = pad_random(X, self.cut)
        x_inp = Tensor(X_pad)
        y = self.labels[key]
        return x_inp, y

class CustomAudioData():
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
        one_dim_signal = self._two_channel(signal)
        resampled_signal = self._resampling(self.resampled_sr, sr, one_dim_signal)
        padded_signal = self._padding(self.target_len, resampled_signal)
        padded_signal = padded_signal.squeeze()
        # spectrogram_signal = self._spectrograms(self.resampled_sr, n_mels = self.mels, n_fft=self.fft, signal=padded_signal)
        return padded_signal, cls

    def _two_channel(self, signal_vec):
        if signal_vec.shape[0] == 2:
            signal_vec = torch.mean(signal_vec,dim=0).unsqueeze(0)
        return signal_vec

    def _resampling(self, resample_sr, oldsr, signal):
        resampler = torchaudio.transforms.Resample(oldsr, resample_sr)
        sample1 = resampler(signal)
        return sample1

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


class Dataset_ASVspoof2019_devNeval(Dataset):
    def __init__(self, list_IDs, base_dir):
        """self.list_IDs	: list of strings (each string: utt key),
        """
        self.list_IDs = list_IDs
        self.base_dir = base_dir
        self.cut = 64600  # take ~4 sec audio (64600 samples)

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        key = self.list_IDs[index]
        X, _ = sf.read(str(self.base_dir / f"flac/{key}.flac"))
        X_pad = pad(X, self.cut)
        x_inp = Tensor(X_pad)
        return x_inp, key
