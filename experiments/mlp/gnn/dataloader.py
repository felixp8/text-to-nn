import numpy as np
import pandas as pd
import h5py
import torch
import torch.nn as nn
import itertools

from pathlib import Path
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch_geometric.data import Data
from torch_geometric.loaders import DataLoader
from typing import Union, Optional, Literal


ACTIVATIONS = [ # TODO: find a good way to do this 
    nn.ReLU,
    nn.Sigmoid,
    nn.Tanh,
    nn.GELU,
    nn.SiLU,
]


def make_dataloader(
    expression_file: Union[str, Path]
    parameter_file: Union[str, Path],
    architecture: nn.Module,
    expression_encoding: Literal["label", "onehot"] = "label",
    activation_encoding: Literal["label", "onehot"] = "onehot",
    depth_encoding: Literal["label", "onehot"] = "onehot",
    splits: list[float] = [0.7, 0.3],
    shuffle: list[bool] = [True, False]
    batch_size: int = 32,
    seed: Optional[int] = None,
):
    rng = np.random.default_rng(seed)
    # load data
    expression_df = pd.read_csv(expression_file)
    with h5py.File(parameter_file, 'r') as h5f:
        parameters = h5f['nn_parameters'][:h5f['counter'][()].item(), :]
    assert expression_df.shape[0] == parameters.shape[0]
    n_samples = expression_df.shape[0]

    # encode expressions as int
    if expression_encoding == "label":
        expr_enc = LabelEncoder()
        expressions = expr_enc.fit_transform(expression_df.expr)
    elif expression_encoding == "onehot":
        expr_enc = OneHotEncoder()
        expressions = expr_enc.fit_transform(expression_df.expr)
    else:
        raise ValueError

    # prepare features
    depth = len(architecture.dims)
    n_nodes = sum(architecture.dims)
    n_edges = sum([architecture.dims[i] * architecture.dims[i+1] for i in range(len(architecture.dims)-1)]) * 2
    # node depths
    node_depths = np.repeat(np.arange(depth), architecture.dims)
    assert node_depths.shape[0] == n_nodes
    # prepare arrs for activations and edges
    node_activations = np.empty((n_nodes,), dtype='object')
    edges = np.empty((n_edges, 2), dtype=int)
    # don't really need to iterate for edges, but might as well?
    depth = 0
    node_start_idx = 0
    edge_start_idx = 0
    for layer in architecture.layers:
        if isinstance(layer, nn.Linear):
            in_features = layer.in_features
            out_features = layer.out_features
            edges[edge_start_idx:(edge_start_idx+in_features*out_features*2):2, :] = node_start_idx + np.array(
                list(itertools.product(range(in_features), range(in_features, in_features + out_features))))
            edges[edge_start_idx+1:(edge_start_idx+1+in_features*out_features*2):2, :] = node_start_idx + np.array(
                list(itertools.product(range(in_features), range(in_features, in_features + out_features))))[:, ::-1]
            depth += 1
        elif type(layer) in ACTIVATIONS:
            node_activations[node_depths == depth] = layer.__class__.__name__
    # change node depths to one hot, optionally
    if depth_encoding == "onehot":
        temp = np.zeros((n_nodes, depth))
        temp[np.arange(n_nodes), node_depths] = 1
        node_features = temp
    else:
        node_features = node_depths
    # encode node activations
    if len(np.unique(node_activations)) == 1:
        pass
    elif activation_encoding == "label":
        act_enc = LabelEncoder()
        node_activations = act_enc.fit_transform(node_activations)
        node_features = np.concatenate([node_features, node_activations], axis=1)
    elif expression_encoding == "onehot":
        act_enc = OneHotEncoder()
        node_activations = act_enc.fit_transform(node_activations)
        node_features = np.concatenate([node_features, node_activations], axis=1)
    else:
        raise ValueError

    base_node_features = torch.from_numpy(node_features)
    base_edge_idx = torch.from_numpy(edges.T)

    data_list = []
    for i in range(n_samples):
        nn.utils.vector_to_parameters(
            torch.from_numpy(parameters[i, :]), 
            architecture.parameters(),
        )
        edge_features = torch.zeros((n_edges, 1))
        node_features = torch.cat([base_node_features, torch.zeros((n_nodes, 1))], dim=1)
        for layer in architecture.layers:
            if isinstance(layer, nn.Linear):
                in_features = layer.in_features
                out_features = layer.out_features
                edge_features[edge_start_idx:(edge_start_idx+in_features*out_features*2):2, :] = layer.weight.t().flatten()
                edge_features[edge_start_idx+1:(edge_start_idx+1+in_features*out_features*2):2, :] = layer.weight.t().flatten()
                depth += 1
                node_features[(node_depths == depth), -1] = layer.bias
        data = Data(
            x=node_features,
            edge_idx=base_edge_idx,
            edge_attr=edge_features,
            y=torch.from_numpy(expressions[i:(i+1),None]),
        )
        data_list.append(data)
    
    ordering = rng.shuffle(len(data_list))
    splits = [0] + np.round(np.cumsum(splits) / np.sum(splits)).astype(int).tolist()
    dataloaders = []
    for i in range(len(splits)-1):
        dataloader = DataLoader([data_list[n] for n in range(splits[i], splits[i+1])], batch_size=batch_size, shuffle=shuffle[i])
        dataloaders.append(dataloader)

    return dataloaders
        
