import numpy as np
import torch
import pytorch_lightning as pl

from numpy import square, abs, sqrt, sin, cos, exp, log

from expression_generator import expr_sampler

def eval_expr(expr: str, inputs: list):
    for n, inp in enumerate(inputs):
        expr = expr.replace(f'i{n}', str(inp))
    return eval(expr)

def validate_expr(expr: str, n_inputs: int, n_samples: int = 100):
    if not any([(f'i{n}' in expr) for n in range(n_inputs)]):
        raise ValueError("Sampled expression is constant")
    for _ in range(n_samples):
        inputs_sampled = np.random.normal(loc=0., scale=10., size=(n_inputs,))
        output = eval_expr(expr, inputs_sampled)
        if output == 'nan':
            raise ValueError(f"Sampled expression {expr} errored for inputs {inputs_sampled.tolist()}")

class ExpressionDataModule(pl.LightningDataModule):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        batch_size: int = 32,
        dataset_size: int = 1024,
        split_ratios: list[float] = [0.6, 0.2, 0.2],
        expr_list: list = [],
        reloadable: bool = False,
        sampler_kwargs: dict = {},
    ):
        assert len(split_ratios) == 3

        super().__init__()

        self.batch_size = batch_size
        self.dataset_size = dataset_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.split_ratios = split_ratios
        self.reloadable = reloadable

        if len(expr_list) == 0:
            expr_list = []
            for i in range(output_dim):
                num_retries = 0
                while num_retries < 10:
                    expr = expr_sampler(input_dim, **sampler_kwargs)
                    try:
                        validate_expr(expr, n_inputs=input_dim, n_samples=1000)
                        break
                    except:
                        num_retries += 1
                if num_retries == 10: # this shouldn't happen?
                    raise ValueError(f"Failed to generate valid expression. Last attempt: {expr}")
                expr_list.append(expr)
            assert len(expr_list) == output_dim
        else:
            assert len(expr_list) == output_dim
            for expr in expr_list:
                validate_expr(expr, n_inputs=input_dim, n_samples=100)
        self.expr_list = expr_list

    def setup(self, stage):
        self.train_size = round(self.dataset_size * self.split_ratios[0])
        self.val_size = round(self.dataset_size * self.split_ratios[1])
        self.test_size = round(self.dataset_size * self.split_ratios[2]) \
            if len(self.split_ratios) > 2 else 0

        if not self.reloadable:
            inputs, targets = self.generate_samples(self.train_size)
            ds = torch.utils.data.TensorDataset(
                torch.from_numpy(inputs).to(torch.get_default_dtype()),
                torch.from_numpy(targets).to(torch.get_default_dtype()),
            )
            self.train = ds
        else:
            self.train = None

        inputs, targets = self.generate_samples(self.val_size)
        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(inputs).to(torch.get_default_dtype()),
            torch.from_numpy(targets).to(torch.get_default_dtype()),
        )
        self.val = ds

        inputs, targets = self.generate_samples(self.test_size)
        ds = torch.utils.data.TensorDataset(
            torch.from_numpy(inputs).to(torch.get_default_dtype()),
            torch.from_numpy(targets).to(torch.get_default_dtype()),
        )
        self.test = ds

    def generate_samples(self, num_samples, input_mean=0., input_std=3.):
        inputs = np.empty((num_samples, self.input_dim))
        targets = np.empty((num_samples, self.output_dim))

        for i in range(num_samples):
            inputs_sampled = np.random.normal(loc=input_mean, scale=input_std, size=(self.input_dim,))
            targets_sampled = np.array([
                eval_expr(expr, inputs_sampled)
                for expr in self.expr_list
            ])
            if np.any(np.isnan(targets_sampled)):
                raise ValueError()
            inputs[i, :] = inputs_sampled
            targets[i, :] = targets_sampled

        return inputs, targets

    def train_dataloader(self):
        if self.reloadable or self.train is None:
            inputs, targets = self.generate_samples(self.train_size)

            ds = torch.utils.data.TensorDataset(
                torch.from_numpy(inputs).to(torch.get_default_dtype()),
                torch.from_numpy(targets).to(torch.get_default_dtype()),
            )
        else:
            ds = self.train
        return torch.utils.data.DataLoader(ds, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val, batch_size=self.batch_size)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test, batch_size=self.batch_size)