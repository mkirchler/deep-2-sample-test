import contextlib
import numpy as np

import freqopttest.util as fot_util
import freqopttest.data as fot_data
import freqopttest.kernel as fot_kernel
import freqopttest.tst as fot_tst

from base import *

class FOTTest(TwoSampleTest):
    '''Base class for tests implemented in freqopttest'''
    def __init__(self):
        pass

    def preprocess(self, X, Y):
        if len(X.shape) > 2:
            X = X.reshape(len(X), -1)
            Y = Y.reshape(len(Y), -1)
        XY = fot_data.TSTData(X, Y)
        return XY
 

class METest_rand(FOTTest):
    '''Mean Embedding test with random locations and median kernel selection'''
    def __init__(self, J=10, alpha=0.05):
        self.J = J
        self.alpha = alpha

    def test(self, X, Y):
        XY = self.preprocess(X, Y)
        
        locations = fot_tst.MeanEmbeddingTest.init_locs_subset(XY, self.J)
        med = fot_util.meddistance(XY.stack_xy(), 1000)
        kernel = fot_kernel.KGauss(med)
        ME = fot_tst.MeanEmbeddingTest(locations, med, alpha=self.alpha)

        result = ME.perform_test(XY)
        p_val = result['pvalue']
        return p_val


class METest_grid(FOTTest):
    '''Mean Embedding test with random locations and grid-search kernel selection'''
    def __init__(self, J=10, split_ratio=0.5, alpha=0.05):
        self.split_ratio = split_ratio
        self.J = J
        self.alpha = alpha

    def test(self, X, Y):
        XY = self.preprocess(X, Y)
        train, test = XY.split_tr_te(tr_proportion=self.split_ratio)

        locations = fot_tst.MeanEmbeddingTest.init_locs_subset(train, self.J)
        med = fot_util.meddistance(train.stack_xy(), 1000)
        gwidth, info = fot_tst.MeanEmbeddingTest.optimize_gwidth(
                train, locations, med**2)
        
        ME = fot_tst.MeanEmbeddingTest(locations, gwidth, alpha=self.alpha)

        result = ME.perform_test(test)
        p_val = result['pvalue']
        return p_val


class METest_full(FOTTest):
    '''Mean Embedding test with optimized locations and grid-search kernel selection'''
    def __init__(self, J=10, split_ratio=0.5, alpha=0.05):
        self.split_ratio = split_ratio
        self.J = J
        self.alpha = alpha

    def test(self, X, Y):
        XY = self.preprocess(X, Y)
        
        train, test = XY.split_tr_te(tr_proportion=self.split_ratio)

        with contextlib.redirect_stdout(None):
            test_locs, gwidth, info = fot_tst.MeanEmbeddingTest.optimize_locs_width(
                    train,
                    self.alpha,
                    n_test_locs=self.J,
                    )

        ME = fot_tst.MeanEmbeddingTest(test_locs, gwidth, alpha=self.alpha)

        result = ME.perform_test(test)
        p_val = result['pvalue']
        return p_val


class MMDTest_med(FOTTest):
    '''MMD test with median distance kernel selection'''
    def __init__(self, n_permute=200, alpha=0.05):
        self.n_permute = n_permute
        self.alpha = alpha

    def test(self, X, Y):
        XY = self.preprocess(X, Y)

        med = fot_util.meddistance(XY.stack_xy(), 1000)
        kernel = fot_kernel.KGauss(med)

        MMD = fot_tst.QuadMMDTest(kernel, n_permute=self.n_permute, alpha=self.alpha)

        result = MMD.perform_test(XY)
        p_val = result['pvalue']
        return p_val


class MMDTest_opt(FOTTest):
    '''MMD test with grid-search kernel selection'''
    def __init__(self, n_permute=200, alpha=0.05, split_ratio=0.5):
        self.n_permute = n_permute
        self.alpha = alpha
        self.split_ratio = split_ratio

    def test(self, X, Y):
        XY = self.preprocess(X, Y)

        train, test = XY.split_tr_te(tr_proportion=self.split_ratio)
        med = fot_util.meddistance(train.stack_xy(), 1000)

        bandwidths = (med**2) * (2.**np.linspace(-4, 4, 20))
        kernels = [fot_kernel.KGauss(width) for width in bandwidths]
        with contextlib.redirect_stdout(None):
            best_i, powers = fot_tst.QuadMMDTest.grid_search_kernel(
                    train, kernels, alpha=self.alpha)
        best_kernel = kernels[best_i]

        MMD = fot_tst.QuadMMDTest(best_kernel, n_permute=self.n_permute, alpha=self.alpha)

        result = MMD.perform_test(test)
        p_val = result['pvalue']
        return p_val


class SCFTest_rand(FOTTest):
    '''SCF test with random frequencies'''
    def __init__(self, J=10, alpha=0.05):
        self.J = J
        self.alpha = alpha

    def test(self, X, Y):
        XY = self.preprocess(X, Y)

        SCF = fot_tst.SmoothCFTest.create_randn(XY, self.J, alpha=self.alpha, seed=1)
        result = SCF.perform_test(XY)
        p_val = result['pvalue']
        return p_val


class SCFTest_grid(FOTTest):
    '''SCF test with grid-search selection'''
    def __init__(self, J=10, split_ratio=0.5, alpha=0.05):
        self.split_ratio = split_ratio
        self.J = J
        self.alpha = alpha

    def test(self, X, Y):
        XY = self.preprocess(X, Y)
        train, test = XY.split_tr_te(tr_proportion=self.split_ratio)

        freqs = np.random.randn(self.J, XY.dim())

        mean_sd = train.mean_std()
        scales = 2.**np.linspace(-4, 4, 30)
        list_gwidth = np.hstack([mean_sd*scales*(XY.dim()**0.5), 2**np.linspace(-8, 8, 20)])
        list_gwidth.sort()
        with contextlib.redirect_stdout(None):
            best_i, powers = fot_tst.SmoothCFTest.grid_search_gwidth(
                    train, freqs, list_gwidth, self.alpha)
        best_width = list_gwidth[best_i]

        SCF = fot_tst.SmoothCFTest(freqs, best_width, self.alpha)

        result = SCF.perform_test(test)
        p_val = result['pvalue']
        return p_val


class SCFTest_full(FOTTest): 
    '''SCF test with optimized kernel and frequency selection'''
    def __init__(self, J=10, split_ratio=0.5, alpha=0.05):
        self.split_ratio = split_ratio
        self.J = J
        self.alpha = alpha

    def test(self, X, Y):
        XY = self.preprocess(X, Y)

        train, test = XY.split_tr_te(tr_proportion=self.split_ratio)

        with contextlib.redirect_stdout(None):
            with contextlib.redirect_stderr(None):
                test_freqs, gwidth, info = fot_tst.SmoothCFTest.optimize_freqs_width(
                        train,
                        self.alpha,
                        n_test_freqs=self.J
                        )
        SCF = fot_tst.SmoothCFTest(test_freqs, gwidth, alpha=self.alpha)
        result = SCF.perform_test(test)
        p_val = result['pvalue']
        return p_val
