import os
import abc
from abc import  abstractmethod

import numpy as np
from scipy.io import wavfile

from tqdm import tqdm

import torch
import torchvision
from torchvision import transforms

ABC = abc.ABCMeta('ABC', (object,), {'__slots__': ()})


#########
######### fill in the paths to your data here
PATH_TO_BIRDS = 'data/CUB_200_2011/images/'
PATH_TO_PLANES = 'data/fgvc-aircraft-2013b/testing/'
PATH_TO_DOGS = 'data/two-dogs/'
PATH_TO_FACES = 'data/faces/'
PATH_TO_AUDIO = 'data/sound/taor/'
#########
#########


# example usage
def get_faces_data(m=20, gray=False, target_shape=(224, 224), cropping=False):
    '''load facial expression images with optional center cropping for kernel tests'''
    if cropping:
        cropping = (762-300, 562-200)
    else:
        cropping = None
    return ImageData(
            c0='positive',
            c1='negative',
            path_to_data=PATH_TO_FACES,
            gray=gray,
            target_shape=target_shape,
            cropping=cropping)


class TestData(ABC):
    '''(Abstract) base class for all data modules
    
    Every subclass needs to implement a .get_data(H0) method, that takes a
    boolean (whether to sample from H0 or H1) and outputs two sets of observations
    from the two distributions to be tested. Output should be two np.ndarray's of
    size (m, *d) where m = #observations/population, d = dimensionality (might be
    non-scalar, e.g. d = (3, 128, 128) for images, d = 10 for 10-dim features)
    Subclasses also need to implement a .test_h0() function that returns a bool
    whether the null hypothesis at a given self.m-value can be evaluated
    '''
    def __init__(self):
        pass

    @abstractmethod
    def test_h0(self):
        pass

    @abstractmethod
    def get_data(self, H0=True):
        pass


class TorchData(TestData):
    '''(Abstract) superclass for torch data reading utility

    Subclasses only need to load data as torch.Tensors into self.c0_data
    and self.c1_data

    # Parameters:
    m (int): number of observations per sample
    '''
    def __init__(self, m=200):
        super(TorchData, self).__init__()
        self.m = m

    def test_h0(self):
        return len(self.c0_data) >= 2*self.m

    def get_data(self, H0=True):
        perm0 = torch.randperm(len(self.c0_data))
        X = self.c0_data[perm0[:self.m]]
        if H0:
            Y = self.c0_data[perm0[self.m:(2*self.m)]]
        else:
            perm1 = torch.randperm(len(self.c1_data))
            Y = self.c1_data[perm1[:self.m]]
        return np.array(X), np.array(Y)


class ImageData(TorchData):
    '''Data object for the natural image experiments

    Data must be ordered into directories named after classes.

    # Parameters:
    c0 (str): name of X-class (must match folder name)
    c1 (str): name of Y-class (must match folder name)
    path_to_data (str): path to root directory of image data; must contain directories with
                        classes c0 and c1
    m (int): number of observations per sample
    gray (bool): whether to transform the data to grayscale
    target_shape (tuple(int, int)): target shape for the images
    cropping (None or tuple(int, int)): apply center cropping to these dimensions before reshaping
    '''
    def __init__(self, c0, c1, path_to_data, m=50, gray=False, target_shape=(224, 224), cropping=None):
        super(ImageData, self).__init__(m=m)
        self.target_shape = target_shape
        self.path = path_to_data
        self.gray = gray
        self.cropping = cropping
        self.load_classes(c0, c1, path_to_data)

    def get_transform(self):
        tfms = [
                transforms.Resize(self.target_shape),
                transforms.ToTensor(),
                ]

        if self.gray:
            tfms.insert(1, transforms.Grayscale())
        else:
            tfms.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        if self.cropping is not None:
            tfms.insert(0, transforms.CenterCrop(self.cropping))

        transform = transforms.Compose(tfms)
        return transform
            
    def load_classes(self, c0, c1, path_to_data):
        transform = self.get_transform()

        tset = torchvision.datasets.ImageFolder(path_to_data, transform=transform)
        loader = torch.utils.data.DataLoader(tset, batch_size=200, shuffle=True, num_workers=8)

        all_class_c0 = []
        all_class_c1 = []
        c0 = tset.classes.index(c0)
        c1 = tset.classes.index(c1)
        for data, target in tqdm(loader):
            all_class_c0.append(data[target==c0])
            all_class_c1.append(data[target==c1])
        self.c0_data = torch.cat(all_class_c0)
        self.c1_data = torch.cat(all_class_c1)


class AMData(TestData):
    '''Data object for the AM audio data experiments

    Data must be preprocessed to 8000 Hz wavefile, which can be done using e.g. ffmpeg with the
    command `ffmpeg -i INFILE.mp3 -ar 8000 OUTFILE.wav'

    # Parameters:
    c0 (str): name of X-class (must match file name)
    c1 (str): name of Y-class (must match file name)
    path_to_data (str): path to root directory of audio data; must contain wav files
                        with classes c0 and c1
    m (int): number of observations per sample
    noise_level (float): std of gaussian noise to be added to the data
    d (int): length of the snippets
    '''
    def __init__(self, c0, c1, path_to_data, m=50, noise_level=1., d=1000):
        super(AMData, self).__init__()
        self.m = m
        self.d = d
        self.noise_level = noise_level
        self._load_process_audio(c0, c1, path_to_data)

    def test_h0(self):
        return True

    def _interp(self, y, factor):
        l = len(y)
        index = np.arange(l)
        xx = np.linspace(0, l-1, l*factor)
        interpolated_y = np.interp(xx, index, y)
        return interpolated_y

    def _amplitude_modulation(self, signal, sample_rate):
        multiple = 3
        transmit_multiple = 5
        offset = 2
        envelope = 1
        upsampled_signal = self._interp(signal, multiple*transmit_multiple)
        carrier_frequency = sample_rate * multiple
        t = np.arange(len(upsampled_signal)) / (sample_rate*multiple*transmit_multiple)
        carrier_signal = np.sin(2*np.pi*carrier_frequency*t)
        am_signal = carrier_signal * (offset + upsampled_signal*envelope)
        return am_signal

    def _load_am(self, path):
        sampling_rate, audio = wavfile.read(path)
        audio = audio.mean(1)
        audio /= audio.std()
        am_signal = self._amplitude_modulation(audio, sampling_rate)
        return am_signal

    def _load_process_audio(self, c0, c1, path_to_data):
        self.c0_data = self._load_am(os.path.join(path_to_data, c0))
        self.c1_data = self._load_am(os.path.join(path_to_data, c1))

    def _select_slices(self, signal):
        ind_Z = np.random.choice(len(signal)-self.d, self.m, replace=True)
        Z = np.empty((self.m, self.d))
        for m in range(self.m):
            start = ind_Z[m]
            end = start + self.d
            Z[m, :] = signal[start:end]
        return Z

    def _load_noisy_sample(self, data):
        Z = self._select_slices(data)
        noise_Z = np.random.normal(0, self.noise_level, size=(self.m, self.d))
        Z += noise_Z
        return Z

    def get_data(self, H0=True):
        X = self._load_noisy_sample(self.c0_data)
        if H0:
            Y = self._load_noisy_sample(self.c0_data)
        else:
            Y = self._load_noisy_sample(self.c1_data)
        return X, Y
