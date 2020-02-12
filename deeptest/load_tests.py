import torch
import torchvision.models as models
import torch.nn as nn

from .embeddingtests import *
from .c2st import *


def get_M5_test(test_type='dfda', device='cpu'):
    '''Load deep 2-sample test with M5 network pretrained on audio data
    
    This is the test used for the experiments on aircraft, birds, and facial expressions

    # Parameters:
    test_type (str): which test statistic to use, possible values are 'dfda', 'dmmd' or 'c2st' 
    device (str): which cpu/cuda device to use
    '''
    weights = torch.load('models/M5weights.pt', map_location='cpu')
    model = M5Net(classes=13)
    model.load_state_dict(weights)
    model = nn.Sequential(model.features, nn.MaxPool1d(4), FlattenLayer(), nn.Tanh())

    reshape = (-1, 1, 1000)
    model_d = 512
    model.eval()
    set_parameters_grad(model, requires_grad=False)

    if test_type == 'dfda':
        T = DFDATest(model, reshape=reshape, device=device)
    elif test_type == 'dmmd':
        T = DMMDTest(model, reshape=reshape, device=device, n_perm=1000)
    elif test_type == 'dmmd_kernel':
        T = DMMDKernelTest(model, reshape=reshape, device=device, kernel_opt=True, n_perm=1000)
    elif test_type == 'c2st':
        T = TransferC2ST(model, model_d, device=device, reshape=reshape)
    else:
        raise NotImplementedError()
    return T
    
def get_resnet_test(test_type='dfda', device='cpu'):
    '''Load deep 2-sample test with resnet-152 model
    
    This is the test used for the experiments on aircraft, birds, and facial expressions

    # Parameters:
    test_type (str): which test statistic to use, possible values are 'dfda', 'dmmd' or 'c2st' 
    device (str): which cpu/cuda device to use
    '''
    model = models.resnet152(pretrained=True)
    model_d = 2048
    model.fc = nn.Tanh()
    set_parameters_grad(model, requires_grad=False)
    model = model.to(device)
    model.eval()

    if test_type == 'dfda':
        T = DFDATest(model, reshape=None, device=device)
    elif test_type == 'dmmd':
        T = DMMDTest(model, reshape=None, device=device, n_perm=1000)
    elif test_type == 'dmmd_kernel':
        T = DMMDKernelTest(model, reshape=None, device=device, kernel_opt=False, n_perm=1000)
    elif test_type == 'c2st':
        T = TransferC2ST(model, model_d, device=device, reshape=None)
    else:
        raise NotImplementedError()
    return T

def get_dogstest_unsupervised(test_type='dfda', device='cpu'):
    '''Load deep 2-sample test with unsupervised pretraining on Stanford dogs

    This is the test used for the unsupervised experiments on the Stanford dogs data set

    # Parameters:
    test_type (str): which test statistic to use, possible values are 'dfda', 'dmmd' or 'c2st' 
    device (str): which cpu/cuda device to use
    '''
    path = 'models/CAEweights.pt'
    model_d = 2048
    weights = torch.load(path, map_location='cpu')
    model = CAE224(d=model_d)
    model.load_state_dict(weights)
    model = nn.Sequential(model.encoder[:-1], FlattenLayer(), nn.Tanh())

    set_parameters_grad(model, requires_grad=False)
    model = model.to(device)
    model.eval()

    if test_type == 'dfda':
        T = DFDATest(model, reshape=None, device=device)
    elif test_type == 'dmmd':
        T = DMMDTest(model, reshape=None, device=device, n_perm=1000)
    elif test_type == 'dmmd_kernel':
        T = DMMDKernelTest(model, reshape=None, device=device, kernel_opt=False, n_perm=1000)
    elif test_type == 'c2st':
        T = TransferC2ST(model, model_d, device=device, reshape=None)
    else:
        raise NotImplementedError()
    return T

def get_dogstest_supervised(test_type='dfda', device='cpu'):
    '''Load deep 2-sample test with supervised pretraining on Stanford dogs

    This is the test used for the supervised experiments on the Stanford dogs data set

    # Parameters:
    test_type (str): which test statistic to use, possible values are 'dfda', 'dmmd' or 'c2st' 
    device (str): which cpu/cuda device to use
    '''
    path = 'models/Supervised_dogs_weights.pt'
    model_d = 2048
    weights = torch.load(path, map_location='cpu')
    model = EncoderClassifier(d=model_d, c_out=118)
    model.load_state_dict(weights)
    model = nn.Sequential(model.features[:-1], nn.Tanh())

    set_parameters_grad(model, requires_grad=False)
    model = model.to(device)
    model.eval()

    if test_type == 'dfda':
        T = DFDATest(model, reshape=None, device=device)
    elif test_type == 'dmmd':
        T = DMMDTest(model, reshape=None, device=device, n_perm=1000)
    elif test_type == 'dmmd_kernel':
        T = DMMDKernelTest(model, reshape=None, device=device, kernel_opt=False, n_perm=1000)
    elif test_type == 'c2st':
        T = TransferC2ST(model, model_d, device=device, reshape=None)
    else:
        raise NotImplementedError()
    return T


#################################
##### Models for retrieval ######
#################################


class EncoderClassifier(nn.Module):
    '''Covolutional classification NN with same feature map as autoencoder'''
    def __init__(self, d=100, c_out=118):
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
    '''Convolutional autoencoder for (224, 224) images'''
    def __init__(self, d=100):
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
        self.decoder = nn.Sequential(
                nn.ConvTranspose2d(d, 360, 3, padding=p),
                nn.BatchNorm2d(360),
                nn.ReLU(inplace=True),
                nn.Upsample(size=(8, 8)),

                nn.ConvTranspose2d(360, 240, 3, padding=p),
                nn.BatchNorm2d(240),
                nn.ReLU(inplace=True),
                nn.Upsample(size=(24, 24)),

                nn.ConvTranspose2d(240, 160, 3, padding=p),
                nn.BatchNorm2d(160),
                nn.ReLU(inplace=True),
                nn.Upsample(size=(56, 56)),

                nn.ConvTranspose2d(160, 80, 3, padding=p),
                nn.BatchNorm2d(80),
                nn.ReLU(inplace=True),
                nn.Upsample(size=(112, 112)),

                nn.ConvTranspose2d(80, 40, 3, padding=p),
                nn.BatchNorm2d(40),
                nn.ReLU(inplace=True),
                nn.Upsample(size=(222, 222)),

                nn.ConvTranspose2d(40, 3, 3, padding=p),
                nn.BatchNorm2d(3),
                nn.Sigmoid()
                )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class MXNet(nn.Module):
    '''Base class for 1-d convolutional networks'''
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
    '''5-layer 1-d convolutional network for audio classification'''
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
