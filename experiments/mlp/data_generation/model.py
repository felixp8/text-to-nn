import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class MLP(nn.Module):
    def __init__(
        self, 
        input_dim: int, 
        hidden_dims: list, 
        output_dim: int, 
        activation: str = "relu", 
        bias: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_dim]

        if activation.lower() == "relu":
            activation = nn.ReLU
        elif activation.lower() == "sigmoid":
            activation = nn.Sigmoid
        elif activation.lower() == "tanh":
            activation = nn.Tanh
        elif activation.lower() == "gelu":
            activation = nn.GELU
        elif activation.lower() == "silu":
            activation == nn.SiLU
        else:
            raise ValueError()

        layerlist = []
        for i in range(len(dims) - 2):
            layerlist.append(nn.Linear(dims[i], dims[i+1], bias=bias))
            layerlist.append(activation())
            layerlist.append(nn.Dropout(dropout))
        layerlist.append(nn.Linear(dims[-2], dims[-1], bias=True))

        self.layers = nn.Sequential(*layerlist)

    def forward(self, x):
        return self.layers(x)
