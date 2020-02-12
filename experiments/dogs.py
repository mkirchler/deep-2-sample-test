'''script for training network on Stanford Dogs data - use train_model and train_model_cae'''
import numpy as np

import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torch import optim
from torch.utils.data.sampler import SubsetRandomSampler

from pytorch_msssim import msssim

from deeptest import FlattenLayer


def train_model(model, epochs=10, bs=64, lr=0.001, device='cpu', save_str=''):
    '''train supervised CNN on Stanford Dogs data'''
    target_shape = 224
    train, val = get_dogs_data(bs=bs, target_shape=target_shape)

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
    for epoch in range(1, epochs+1):
        print('starting epoch %d/%d' % (epoch, epochs))
        train_one_epoch(model, opt, train, device=device)
        val_acc = evaluate(model, val, device=device)
        if epoch % 5 == 0:
            torch.save({'model':model, 'val_acc':val_acc}, save_str % epoch)

def train_model_cae(model, epochs=10, bs=128, lr=0.001, device='cpu', save_str=''):
    '''train unsupervised CNN on Stanford Dogs data'''
    target_shape = 224
    train, val = get_dogs_data(bs=bs, target_shape=target_shape, normalize=False)

    opt = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(1, epochs+1):
        print('starting epoch %d/%d' % (epoch, epochs))
        train_one_epoch_cae(model, opt, train, device=device)
        val_acc = evaluate_cae(model, val, device=device)
        if epoch % 5 == 0:
            torch.save({'model':model, 'val_acc':val_acc}, save_str % epoch)

def train_one_epoch_cae(model, optimizer, loader, device='cpu'):
    model.train()
    running_loss = 0.
    for i, (data, _) in enumerate(loader):
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = -msssim(output, data, normalize=True)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if i % 5 == 0:
            print('running loss@%d: %.4f' % (i, running_loss/(i+1)))

def train_one_epoch(model, optimizer, loader, device='cpu'):
    model.train()
    running_loss = 0.
    criterion = nn.CrossEntropyLoss()
    for i, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if i % 5 == 0:
            print('running loss@%d: %.4f' % (i, running_loss/(i+1)))

def evaluate_cae(model, loader, device='cpu'):
    model.eval()
    full_loss = 0.
    with torch.no_grad():
        for data, _ in loader:
            data = data.to(device)
            output = model(data)
            loss = -msssim(output, data, normalize=True)
            full_loss += len(data) * loss
        full_loss = full_loss / len(loader.sampler.indices)
    print('loss: %.4f' % (full_loss))
    return full_loss


def evaluate(model, loader, device='cpu'):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
        val_accuracy = correct / len(loader.sampler.indices)
    print('validation accuracy: %.4f' % (val_accuracy))
    return val_accuracy


def get_dogs_data(bs=16, target_shape=224, train_ratio=0.8, normalize=True):
    path = 'data/dogs-without2'
    transform_tr = transforms.Compose([
        transforms.Resize((target_shape, target_shape)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        transforms.RandomRotation(5),
        transforms.ToTensor(),
        ])
    transform_val = transforms.Compose([
        transforms.Resize((target_shape, target_shape)),
        transforms.ToTensor(),
        ])
    if normalize:
        transform_tr = transforms.Compose([
            transform_tr,
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        transform_val = transforms.Compose([
            transform_val,
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    tset = torchvision.datasets.ImageFolder(path, transform=transform_tr)
    vset = torchvision.datasets.ImageFolder(path, transform=transform_val)
    n = len(tset)
    n_train = int(n*train_ratio)
    indices = np.arange(n)
    np.random.shuffle(indices)
    ind_tr, ind_val = indices[:n_train], indices[n_train:]
    sampler_tr = SubsetRandomSampler(ind_tr)
    sampler_val = SubsetRandomSampler(ind_val)

    loader_train = torch.utils.data.DataLoader(tset, batch_size=bs, sampler=sampler_tr, num_workers=12)
    loader_val = torch.utils.data.DataLoader(vset, batch_size=bs, sampler=sampler_val, num_workers=12)
    
    return loader_train, loader_val


class EncoderClassifier(nn.Module):
    def __init__(self, d=2048, c_out=118):
        super(EncoderClassifier, self).__init__()
        self.features = nn.Sequential(
                nn.Conv2d(3, 40, 3, padding=0),
                nn.BatchNorm2d(40),
                nn.ReLU(True),
                nn.MaxPool2d(2),

                nn.Conv2d(40, 80, 3, padding=0),
                nn.BatchNorm2d(80),
                nn.ReLU(True),
                nn.MaxPool2d(2),

                nn.Conv2d(80, 160, 3, padding=0),
                nn.BatchNorm2d(160),
                nn.ReLU(True),
                nn.MaxPool2d(2),

                nn.Conv2d(160, 240, 3, padding=0),
                nn.BatchNorm2d(240),
                nn.ReLU(True),
                nn.MaxPool2d(2),

                nn.Conv2d(240, 360, 3, padding=0),
                nn.BatchNorm2d(360),
                nn.ReLU(True),
                nn.MaxPool2d(2),

                nn.Conv2d(360, d, 3, padding=0),
                nn.BatchNorm2d(d),
                nn.ReLU(True),
                nn.MaxPool2d(2),

                FlattenLayer(),

                nn.Linear(d, c_out),
                )
    def forward(self, x):
        return self.features(x)


class CAE224(nn.Module):
    def __init__(self, d=2048):
        super(CAE224, self).__init__()
        p = 0
        self.encoder = nn.Sequential(
                nn.Conv2d(3, 40, 3, padding=p),
                nn.BatchNorm2d(40),
                nn.ReLU(inplace=True), 
                nn.MaxPool2d(2),
                
                nn.Conv2d(40, 80, 3, padding=p),
                nn.BatchNorm2d(80),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                nn.Conv2d(80, 160, 3, padding=p),
                nn.BatchNorm2d(160),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                
                nn.Conv2d(160, 240, 3, padding=p),
                nn.BatchNorm2d(240),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                nn.Conv2d(240, 360, 3, padding=p),
                nn.BatchNorm2d(360),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),

                nn.Conv2d(360, d, 3, padding=p),
                nn.BatchNorm2d(d),
                nn.MaxPool2d(2),
                nn.Tanh(),
                )
        p2 = 0
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(d, 360, 3, padding=p2),
                nn.BatchNorm2d(360),
                nn.ReLU(inplace=True),
                nn.Upsample(size=(8, 8)),

                nn.ConvTranspose2d(360, 240, 3, padding=p2),
                nn.BatchNorm2d(240),
                nn.ReLU(inplace=True),
                nn.Upsample(size=(24, 24)),

                nn.ConvTranspose2d(240, 160, 3, padding=p2),
                nn.BatchNorm2d(160),
                nn.ReLU(inplace=True),
                nn.Upsample(size=(56, 56)),

                nn.ConvTranspose2d(160, 80, 3, padding=p2),
                nn.BatchNorm2d(80),
                nn.ReLU(inplace=True),
                nn.Upsample(size=(112, 112)),

                nn.ConvTranspose2d(80, 40, 3, padding=p2),
                nn.BatchNorm2d(40),
                nn.ReLU(inplace=True),
                nn.Upsample(size=(222, 222)),

                nn.ConvTranspose2d(40, 3, 3, padding=p2),
                nn.BatchNorm2d(3),
                nn.Sigmoid()
                )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

