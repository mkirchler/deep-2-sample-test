import numpy as np
from scipy import stats

import torch
import torch.nn as nn
from torch import optim

from .base import TwoSampleTest
from .utils import Lambda, FlattenLayer, set_parameters_grad


class TransferC2ST(TwoSampleTest):
    '''Transfer Classifier two-sample test

    # Parameters:
    pretrained_model (nn.Module): pytorch feature extraction neural network
    model_d (int): size of feature embedding of pretrained_model
    device (str): which cuda/cpu device to use
    reshape (None or tuple): if not None, reshape data into this format before passing to pretrained_model
    '''
    def __init__(self, pretrained_model, model_d, device='cpu', reshape=None):
        super(TransferC2ST, self).__init__()
        self.pre_model = pretrained_model.to(device)
        self.pre_model.eval()
        self.model_d = model_d
        self.device = device
        self.reshape = reshape
        self.epochs = 100

    def _split_data(self, X, Y):
        perm_X = torch.randperm(self.n)
        perm_Y = torch.randperm(self.m)
        X_tr, X_te = X[perm_X[:(self.n//2)]], X[perm_X[(self.n//2):]]
        Y_tr, Y_te = Y[perm_Y[:(self.m//2)]], Y[perm_Y[(self.m//2):]]
        return (X_tr, Y_tr), (X_te, Y_te)

    def _preprocess(self, X, Y):
        X, Y = torch.Tensor(X), torch.Tensor(Y)
        if not self.reshape is None:
            X = X.view(*self.reshape)
            Y = Y.view(*self.reshape)
        return X, Y

    def _load_model(self):
        self.model = nn.Sequential(nn.Linear(self.model_d, 1), nn.Sigmoid())

    def _train_model(self, X, Y):
        Z = torch.cat((X, Y))
        labels = torch.cat((torch.zeros(len(X)), torch.ones(len(Y))))
        with torch.no_grad():
            Z_feats = self.pre_model(Z.to(self.device)).cpu()
        
        tset = torch.utils.data.TensorDataset(Z_feats, labels)
        loader = torch.utils.data.DataLoader(tset, batch_size=64, shuffle=True)

        self.model.train()
        loss_func = nn.BCELoss()

        opt = optim.Adam(self.model.parameters())
        for ep in range(self.epochs):
            for i, (data, target) in enumerate(loader):
                opt.zero_grad()
                out = self.model(data)
                loss = loss_func(out.flatten(), target.flatten())
                loss.backward()
                opt.step()

    def _eval_model(self, X, Y):
        self.model.eval()
        with torch.no_grad():
            feats_X = self.pre_model(X.to(self.device)).cpu()
            feats_Y = self.pre_model(Y.to(self.device)).cpu()
            pred_X = self.model(feats_X).numpy()
            pred_Y = self.model(feats_Y).numpy()
            correct_X = (pred_X < 0.5)
            correct_Y = (pred_Y >= 0.5)
            acc = np.concatenate((correct_X, correct_Y)).mean()
            return acc
 
    def _compute_p_value(self, accuracy):
        p_val = 1. - stats.norm.cdf(accuracy, loc=0.5, scale=np.sqrt(0.25/self.n))
        return p_val

    def test(self, X, Y):
        '''Perform two-sample test between X & Y and return p-value

        # Parameters:
        X (np.array): sample X
        Y (np.array): sample Y
        '''
        self.n, self.m = len(X), len(Y)
        assert self.n == self.m
        self.d = X.shape[1]
        X, Y = self._preprocess(X, Y)
        (X_tr, Y_tr), (X_te, Y_te) = self._split_data(X, Y)

        self._load_model()

        self._train_model(X_tr, Y_tr)

        accuracy = self._eval_model(X_te, Y_te)
        p_val = self._compute_p_value(accuracy)
        return p_val
