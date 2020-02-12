from scipy import stats
import numpy as np

from sklearn.decomposition import PCA

import torch

from .base import *
from .kerneltests import MMDTest_med, MMDTest_opt


class EmbeddingTest(TwoSampleTest):
    '''Base Class for the embedding-based two-sample tests

    # Parameters
    model (nn.Module): pytorch feature extraction neural network
    reshape (None or tuple): if not None, reshape data into this format before passing to model
    device (str): which cuda/cpu device to use
    '''
    def __init__(self, model, reshape=None, device=None):
        super(EmbeddingTest, self).__init__()
        if device is None:
            use_cuda = torch.cuda.is_available()
            self.device = torch.device('cuda' if use_cuda else 'cpu')
        else:
            self.device = device
        self.reshape = reshape
        self.eps_ridge = 1e-8    # add ridge to Covariance for numerical stability
        self.model = model.to(self.device)
        self.model.eval()

    def _pca(self, data):
        d = np.sqrt(len(data)/2.).round().astype(int)
        pca = PCA(n_components=d)
        return pca.fit_transform(data)

    def _preprocess(self, X, Y):
        X, Y = torch.Tensor(X), torch.Tensor(Y)
        if not self.reshape is None:
            X = X.view(*self.reshape)
            Y = Y.view(*self.reshape)
        X = X.to(self.device)
        Y = Y.to(self.device)
        return X, Y
 
 
class DMMDTest(EmbeddingTest):
    '''Deep Maximum Mean Discrepancy Test

    # Parameters
    model (nn.Module): pytorch feature extraction neural network
    reshape (None or tuple): if not None, reshape data into this format before passing to model
    device (str): which cuda/cpu device to use
    n_perm (int): how many permutations to draw for estimating the null hypothesis
    '''
    def __init__(self, model, reshape=None, device=None, n_perm=1000):
        super(DMMDTest, self).__init__(model=model, reshape=reshape, device=device)
        self.n_perm = n_perm
        
    def _compute_mmd(self, features_X, features_Y):
        mean_fX = features_X.mean(0)
        mean_fY = features_Y.mean(0)
        D = mean_fX - mean_fY
        statistic = np.linalg.norm(D)**2
        return statistic

    def _compute_p_value(self, features_X, features_Y):
        stat = self._compute_mmd(features_X, features_Y)

        n, m = len(features_X), len(features_Y)
        l = n + m
        features_Z = np.vstack((features_X, features_Y))

        # compute null samples
        resampled_vals = np.empty(self.n_perm)
        for i in range(self.n_perm):
            index = np.random.permutation(l)
            feats_X, feats_Y = features_Z[index[:n]], features_Z[index[n:]]
            resampled_vals[i] = self._compute_mmd(feats_X, feats_Y)
        resampled_vals.sort()
        
        p_val = np.mean(stat < resampled_vals)
        return p_val

    def test(self, X, Y):
        with torch.no_grad():
            X, Y = self._preprocess(X, Y)
            
            features_X = self.model(X).cpu().numpy()
            features_Y = self.model(Y).cpu().numpy()

        return self._compute_p_value(features_X, features_Y)


class DFDATest(EmbeddingTest):
    '''Deep Fisher Discriminant Analysis Test

    # Parameters:
    model (nn.Module): pytorch feature extraction neural network
    reshape (None or tuple): if not None, reshape data into this format before passing to model
    device (str): which cuda/cpu device to use
    '''
    def __init__(self, model, reshape=None, device=None):
        super(DFDATest, self).__init__(model=model, reshape=reshape, device=device)

    def _compute_p_value(self, features_X, features_Y):
        n, d = features_X.shape
        m = len(features_Y)
        mean_fX = features_X.mean(0)
        mean_fY = features_Y.mean(0)
        D = mean_fX - mean_fY

        all_features = np.concatenate([features_X, features_Y])
        Cov_D = (1./n + 1./m) * np.cov(all_features.T) + self.eps_ridge * np.eye(d)

        statistic = D.dot(np.linalg.solve(Cov_D, D))
        p_val = 1. - stats.chi2.cdf(statistic, d)
        return p_val
    
    def test(self, X, Y):
        with torch.no_grad():
            X, Y = self._preprocess(X, Y)
            self.model = self.model.to(self.device)
            
            features_X = self.model(X).cpu().numpy()
            features_Y = self.model(Y).cpu().numpy()

            feats_Z = np.concatenate((features_X, features_Y))
            feats_Z = self._pca(feats_Z)
            features_X = feats_Z[:len(features_X)]
            features_Y = feats_Z[len(features_X):]

        return self._compute_p_value(features_X, features_Y)


class DMMDKernelTest(EmbeddingTest):
    '''Deep Maximum Mean Discrepancy with Gaussian Kernel Test

    # Parameters
    model (nn.Module): pytorch feature extraction neural network
    reshape (None or tuple): if not None, reshape data into this format before passing to model
    device (str): which cuda/cpu device to use
    n_perm (int): how many permutations to draw for estimating the null hypothesis
    opt (bool): whether to select kernel width via data split & optimizing or via the median method
    '''
    def __init__(self, model, reshape=None, device=None, n_perm=1000, kernel_opt=True):
        super(DMMDKernelTest, self).__init__(model=model, reshape=reshape, device=device)
        if kernel_opt:
            self.mmd_test = MMDTest_opt(n_permute=n_perm)
        else:
            self.mmd_test = MMDTest_med(n_permute=n_perm)

    def test(self, X, Y):
        with torch.no_grad():
            X, Y = self._preprocess(X, Y)
            
            features_X = self.model(X).cpu().numpy()
            features_Y = self.model(Y).cpu().numpy()
                
        p_val = self.mmd_test.test(features_X, features_Y)
        return p_val
