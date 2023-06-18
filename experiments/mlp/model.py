import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, activation: str = "relu", bias=True):
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_dim]

        if activation == "relu":
            activation = nn.ReLU
        elif activation == "sigmoid":
            activation = nn.Sigmoid
        elif activation == "tanh":
            activation = nn.Tanh
        elif activation == "gelu":
            activation = nn.GELU
        else:
            raise ValueError()

        layerlist = []
        for i in range(len(dims) - 2):
            layerlist.append(nn.Linear(dims[i], dims[i+1], bias=bias))
            layerlist.append(activation())
        layerlist.append(nn.Linear(dims[-2], dims[-1], bias=True))

        self.layers = nn.Sequential(*layerlist)

    def forward(self, x):
        return self.layers(x)