import abc
from abc import abstractmethod
import numpy as np
from tqdm import tqdm

ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()})


class TestPipe:
    '''Pipeline class to fully evaluate a test on a given data set over n_runs

    # Parameters:
    data (TestData): data object to be tested on
    test (TwoSampleTest): test object to be tested
    alpha (float): significance level in (0, 1)
    n_runs (int): number of runs for evaluation
    '''
    def __init__(self, data, test, n_runs=100, alpha=0.05):
        self.data = data
        self.test = test
        self.alpha = alpha
        self.n_runs = n_runs

    def evaluate_test(self):
        '''run the self.test on self.data for self.n_runs for significance value self.alpha'''
        p_values_H0 = -np.ones(self.n_runs)
        p_values_H1 = -np.ones(self.n_runs)
        for run in tqdm(range(self.n_runs)):
            # enough observations for drawing from H0?
            if self.data.test_h0():
                p_values_H0[run] = self.test.test(*self.data.get_data(H0=True))

            p_values_H1[run] = self.test.test(*self.data.get_data(H0=False))

        if self.data.test_h0():
            T1ER = (p_values_H0 < self.alpha).mean()
        else:
            T1ER = np.nan
        T2ER = 1 - (p_values_H1 < self.alpha).mean()
        return {'T1ER': T1ER, 'T2ER': T2ER}, p_values_H0, p_values_H1, self.alpha


def benchmark_pipe(data, test, data_params=None, n_runs=100, alpha=0.05, verbose=True):
    '''Evaluate TestPipe over different data parameters

    # Parameters:
    data (TestData): data object to be tested on
    test (TwoSampleTest): test object to be tested
    data_params (tuple(str, list)): tuple of key and values, where data_params[0]
                is the name of the parameter (e.g. 'm'), data_params[1] is a list
                of values to evaluate
    alpha (float): significance level in (0, 1)
    n_runs (int): number of runs for evaluation
    '''
    results = []

    if data_params:
        param = data_params[0]
        values = data_params[1]
    else:
        values = [0]

    for value in values:
        if data_params:
            if verbose:
                print('starting %s = %s' % (param, value))
            setattr(data, param, value)

        try:
            errs = TestPipe(data, test, alpha=alpha, n_runs=n_runs).evaluate_test()[0]
        except Exception as e:
            print('throwing exception', e)
            errs = {'T1ER':np.nan, 'T2ER':np.nan}

        results.append((value, errs))
        if verbose:
            print(value, errs)
                
    return results


class TwoSampleTest(ABC):
    '''Base class for all testing procedures
    
    Every subclass needs to implement the .test(X, Y) method, that takes in two sets
    of observations X and Y and outputs a (scalar) p-value
    '''
    def __init__(self):
        pass

    @abstractmethod
    def test(self, X, Y):
        pass

    def reset(self):
        pass
