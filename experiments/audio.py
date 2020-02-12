'''script for training network for audio data - just use 'train_model' function'''
import os
import numpy as np
from scipy.io import wavfile

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim

from deeptest import FlattenLayer


DATA_DIR = 'data/sound/taor'

# train all but songs 04 & 05
OTHER_PATHS = [os.path.join(DATA_DIR, song) for song in
    ['01 Gramatik - Brave Men.wav',
     '02 Gramatik - Torture.wav',
     '03 Gramatik - Bluestep (Album Version).wav',
     "06 Gramatik - You Don't Understand.wav",
     '07 Gramatik - Obviously.wav',
     '08 Gramatik - Control Room Before You.wav',
     '09 Gramatik - Prime Time.wav',
     '10 Gramatik - Get A Grip Feat. Gibbz.wav',
     "11 Gramatik - Just Jammin' NYC.wav",
     '12 Gramatik - Expect Us.wav',
     '13 Gramatik - Faraway.wav',
     '14 Gramatik - No Turning Back.wav',
     "15 Gramatik - It's Just A Ride.wav"]
    ]

 
def train_model(device='cpu', epochs=50, add_noise=1, m_train=10000, bs=128, lr=0.001):
    '''train M5 net on audio data

    # Parameters
    device (str): which cpu/cuda device to use
    epochs (int): number of epochs to train
    add_noise (float): noise level added during training onto data
    m_train (int): number of training samples
    bs (int): batch-size
    lr (float): learning rate
    '''
    train, val = load_audio_data(m_train=m_train, m_val=1000, bs=bs)
    n_classes = 13

    model = M5Net(n_classes).to(device)
    
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    for epoch in range(1, epochs+1):
        print('starting epoch %d/%d' % (epoch, epochs))
        train_one_epoch(model, opt, train, device=device, add_noise=add_noise)
        evaluate(model, val, device=device)
    return model
    
def train_one_epoch(model, opt, loader, device='cpu', add_noise=1):
    model.train()
    running_loss = 0.
    for i, (data, target) in enumerate(loader):
        opt.zero_grad()
        data, target = data.to(device), target.to(device)
        
        if add_noise:
            n_data, _, d = data.shape
            noise_levels = add_noise*torch.rand(n_data, device=device)
            noise = torch.Tensor(n_data, d).to(device)
            noise.normal_()
            noise.t().mul_(noise_levels)
            data = data + noise.t().contiguous().view(n_data, 1, d)
        
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        opt.step()
        running_loss += loss.item()
        if i % 250 == 0:
            print('batch %d, running loss: %.3f' % (i, running_loss / (i+1)))

def evaluate(model, loader, device='cpu'):
    model.eval()
    full_loss = 0.
    correct = 0.
    n = 0
    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)
            pred = output.argmax(dim=-1)
            full_loss += loss.item()
            correct += (pred.flatten().to(torch.float32) == target.flatten().to(torch.float32)).sum().item()
            n += len(target)
        
        accuracy = correct / float(n)
    print('val loss: %4f, val accuracy: %.4f' % (full_loss, accuracy))

def load_audio_data(m_train, m_val, bs=128, d=1000):
    '''load training data from files into torch loaders'''
    validation_noise = 0.1
    paths = OTHER_PATHS

    train = []
    val = []
    for i, path in enumerate(paths):
        snippets_train = extract_snippets(path, m=m_train, d=d, noise_level=0.)
        labels_train = i * torch.ones(m_train, dtype=torch.long)
        train.append((snippets_train, labels_train))
        snippets_val = extract_snippets(path, m=m_val, d=d, noise_level=validation_noise)
        labels_val = i * torch.ones(m_val, dtype=torch.long)
        val.append((snippets_val, labels_val))
    Z_train = torch.cat([torch.from_numpy(a.astype(np.float32)) for a, b in train]).view(-1, 1, d)
    labels_train = torch.cat([b for a, b in train])
    Z_val = torch.cat([torch.from_numpy(a.astype(np.float32)) for a, b in val]).view(-1, 1, d)
    labels_val = torch.cat([b for a, b in val])

    train = torch.utils.data.TensorDataset(Z_train, labels_train)
    loader_train = torch.utils.data.DataLoader(train, batch_size=bs, shuffle=True)
    val = torch.utils.data.TensorDataset(Z_val, labels_val)
    loader_val = torch.utils.data.DataLoader(val, batch_size=bs, shuffle=False)

    return loader_train, loader_val


class MXNet(nn.Module):
    def __init__(self, classes=1):
        super(MXNet, self).__init__()
        if classes == 2:
            classes = 1
        self.classes = classes

        self.classifier = nn.Sequential(
                nn.ReLU(True),
                nn.MaxPool1d(4),
                FlattenLayer(),
                nn.Linear(512, classes)
                )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        if self.classes == 1:
            x = torch.sigmoid(x)
        return x


class M5Net(MXNet):
    def __init__(self, classes=1):
        super(M5Net, self).__init__(classes=classes)
        self.features = nn.Sequential(
                # kernel_size = 80 originally
                nn.Conv1d(in_channels=1, out_channels=128, kernel_size=20, stride=4),
                nn.BatchNorm1d(128),
                nn.ReLU(True),
                nn.MaxPool1d(4),

                nn.Conv1d(128, 128, 3),
                nn.BatchNorm1d(128),
                nn.ReLU(True),
                nn.MaxPool1d(4),

                nn.Conv1d(128, 256, 3),
                nn.BatchNorm1d(256),
                nn.ReLU(True),
                nn.MaxPool1d(4),

                nn.Conv1d(256, 512, 3),
                nn.BatchNorm1d(512),
                )


# data preprocessing

def extract_snippets(path_to_file, m=100, d=1000, noise_level=0.):
    sampling_rate, audio = wavfile.read(path_to_file)
    audio = audio.mean(1)
    audio /= audio.std()
    am_signal = amplitude_modulation(audio, sampling_rate)
    ind = np.random.choice(len(am_signal)-d, m, replace=True)
    Z = np.empty((m, d))
    for i in range(m):
        start = ind[i]
        end = start + d
        Z[i, :] = am_signal[start:end]

    if noise_level:
        noise = np.random.normal(0, noise_level, size=(m, d))
        Z += noise
    return Z

def amplitude_modulation(signal, fs, multiple=3, transmit_multiple=5, offset=2, envelope=1):
    upsampled_signal = interp(signal, multiple*transmit_multiple)

    carrier_frequency = fs*multiple

    t = np.arange(len(upsampled_signal)) / (fs*multiple*transmit_multiple)

    carrier_signal = np.sin(2*np.pi*carrier_frequency*t)
    
    am_signal = carrier_signal * (offset + upsampled_signal*envelope)
    return am_signal

def interp(y, factor):
    index = np.arange(len(y))
    xx = np.linspace(index[0], index[-1], len(y)*factor)
    interpolated_y = np.interp(xx, index, y)
    return interpolated_y
