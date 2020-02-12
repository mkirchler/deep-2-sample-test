import torch.nn as nn

class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func
    def forward(self, x):
        return self.func(x)


class FlattenLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x.view(len(x), -1)


def set_parameters_grad(model, requires_grad):
    '''update requires_grad for all paramters in model'''
    for param in model.parameters():
        param.requires_grad = requires_grad
