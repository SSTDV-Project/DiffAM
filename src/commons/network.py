import torch
import torch.nn as nn

from .module import GaussianFourierFeatureTransform

class Linear(nn.Module):
    def __init__(self, in_channels, out_channels, activation=None, normalization=None):
        super().__init__()

        layers = [
            nn.Linear(in_channels, out_channels),
        ]
        if activation is not None:
            layers.append(activation)
        if normalization is not None:
            layers.append(normalization)

        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class Linear2(nn.Module):
    def __init__(self, in_channels, out_channels, activation=None, normalization=None):
        super().__init__()

        layers = [
            nn.Linear(in_channels, out_channels),
        ]
        if normalization is not None:
            layers.append(normalization)
        if activation is not None:
            layers.append(activation)

        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class BaseMLP(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_layers, hidden_neurons):
        super().__init__()

        layers = []
        for i in range(hidden_layers):
            in_i = in_channels if i == 0 else hidden_neurons
            out_i = out_channels if i == hidden_layers-1 else hidden_neurons
            activation = nn.Sigmoid() if i == hidden_layers-1 else nn.ReLU()
            normalization = None if i == hidden_layers-1 else nn.LayerNorm(hidden_neurons)
            layers.append(Linear(in_i, out_i, activation, normalization))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class BaseMLPUnbounded(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_layers, hidden_neurons):
        super().__init__()

        layers = []
        for i in range(hidden_layers):
            in_i = in_channels if i == 0 else hidden_neurons
            out_i = out_channels if i == hidden_layers-1 else hidden_neurons
            activation = None if i == hidden_layers-1 else nn.ReLU()
            normalization = None if i == hidden_layers-1 else nn.LayerNorm(hidden_neurons)
            layers.append(Linear(in_i, out_i, activation, normalization))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class BaseMLP2(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_layers, hidden_neurons):
        super().__init__()

        layers = []
        for i in range(hidden_layers):
            in_i = in_channels if i == 0 else hidden_neurons
            out_i = out_channels if i == hidden_layers-1 else hidden_neurons
            activation = None if i == hidden_layers-1 else nn.ReLU() # nn.LeakyReLU(0.2)
            normalization = None # if i == hidden_layers-1 else nn.LayerNorm(hidden_neurons)
            layers.append(Linear(in_i, out_i, activation, normalization))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

class StepNLP(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_layers, hidden_neurons):
        super().__init__()

        self.model = BaseMLP(in_channels, out_channels, hidden_layers, hidden_neurons)
        
        self.bias = nn.Parameter(torch.zeros(out_channels))
    
    def parameters_minimized(self):
        return self.model.parameters()
    
    def parameters_maximized(self):
        return [self.bias]
    
    def step_regularizer(self, scale):
        return torch.exp(-(((self.bias / scale) ** 2).sum()))
    
    def forward(self, x):
        val = self.model(x)
        return val + torch.heaviside(x,x) * self.bias

class SmoothStepNLP(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_layers, hidden_neurons):
        super().__init__()

        self.model = BaseMLP(in_channels, out_channels, hidden_layers, hidden_neurons)
        
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.log_stiffness = nn.Parameter(torch.zeros(in_channels))
    
    def parameters_minimized(self):
        return self.model.parameters()
    
    def parameters_maximized(self):
        return [self.bias, self.log_stiffness]
    
    def step_regularizer(self, scale):
        return torch.exp(-(((self.bias / scale) ** 2).sum())) + torch.exp(-self.log_stiffness / scale).sum()
    
    def forward(self, x):
        val = self.model(x)
        return val + torch.tanh(x * torch.exp(self.log_stiffness)) * self.bias

class TwoNLP(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_layers, hidden_neurons):
        super().__init__()
        self.in_channels = in_channels

        self.model_pos = BaseMLP(in_channels, out_channels, hidden_layers, hidden_neurons)
        self.model_neg = BaseMLP(in_channels, out_channels, hidden_layers, hidden_neurons)
        self.log_slope = nn.Parameter(torch.zeros(out_channels))
    
    def parameters_minimized(self):
        return (*self.model_pos.parameters(), *self.model_neg.parameters())
    
    def parameters_maximized(self):
        return [self.log_slope]
    
    def step_regularizer(self, scale):
        device = next(self.model_pos.parameters()).device
        dtype = next(self.model_pos.parameters()).dtype
        zero = torch.zeros(self.in_channels, dtype=dtype, device=device)
        diff = self.model_pos(zero) - self.model_neg(zero)
        slope = nn.functional.softplus(self.log_slope)
        return torch.exp(-(((diff / slope) ** 2).sum())) * slope
    
    def forward(self, x):
        val_pos = self.model_pos(x)
        val_neg = self.model_neg(x)
        val = torch.where(x<0, val_neg, val_pos)
        return val

class FFNLP(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_layers, hidden_neurons, ff_scale = 10):
        super().__init__()

        self.model = nn.Sequential(
            GaussianFourierFeatureTransform(in_channels, hidden_neurons // 2, ff_scale),
            BaseMLP(hidden_neurons, out_channels, hidden_layers, hidden_neurons)
        )
    
    def forward(self, x):
        return self.model(x)

class FFNLPUnbounded(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_layers, hidden_neurons, ff_scale = 10):
        super().__init__()

        self.model = nn.Sequential(
            GaussianFourierFeatureTransform(in_channels, hidden_neurons // 2, ff_scale),
            BaseMLPUnbounded(hidden_neurons, out_channels, hidden_layers, hidden_neurons)
        )
    
    def forward(self, x):
        return self.model(x)

class FFNLP2(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_layers, hidden_neurons, ff_scale = 10):
        super().__init__()

        self.model = nn.Sequential(
            GaussianFourierFeatureTransform(in_channels, hidden_neurons // 2, ff_scale),
            BaseMLP2(hidden_neurons, out_channels, hidden_layers, hidden_neurons)
        )
    
    def forward(self, x):
        return self.model(x)
    
class StepFFNLP(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_layers, hidden_neurons, ff_scale = 10):
        super().__init__()

        self.model = nn.Sequential(
            GaussianFourierFeatureTransform(in_channels, hidden_neurons // 2, ff_scale),
            BaseMLP(hidden_neurons, out_channels, hidden_layers, hidden_neurons)
        )
        
        self.bias = nn.Parameter(torch.zeros(out_channels))

    def parameters_minimized(self):
        return self.model.parameters()
    
    def parameters_maximized(self):
        return [self.bias]
    
    def step_regularizer(self, scale):
        return torch.exp(-(((self.bias / scale) ** 2).sum()))
    
    def forward(self, x):
        val = self.model(x)
        return val + torch.heaviside(x,x) * self.bias

class SmoothStepFFNLP(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_layers, hidden_neurons, ff_scale = 10):
        super().__init__()

        self.model = nn.Sequential(
            GaussianFourierFeatureTransform(in_channels, hidden_neurons // 2, ff_scale),
            BaseMLP(hidden_neurons, out_channels, hidden_layers, hidden_neurons)
        )
        
        self.bias = nn.Parameter(torch.zeros(out_channels))
        self.log_stiffness = nn.Parameter(torch.zeros(in_channels))
    
    def parameters_minimized(self):
        return self.model.parameters()
    
    def parameters_maximized(self):
        return [self.bias, self.log_stiffness]
    
    def step_regularizer(self, scale):
        return torch.exp(-(((self.bias / scale) ** 2).sum())) + torch.exp(-self.log_stiffness).sum()
    
    def forward(self, x):
        val = self.model(x)
        return val + torch.tanh(x * torch.exp(self.log_stiffness)) * self.bias
