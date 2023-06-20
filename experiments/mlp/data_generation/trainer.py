import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

from model import MLP

class MLPWrapper(pl.LightningModule):
    def __init__(self, input_dim: int, hidden_dims: list, output_dim: int, activation: str = "relu", bias=True):
        super().__init__()
        self.mlp = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            activation=activation,
            bias=bias,
        )

    def forward(self, x):
        return self.mlp(x)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-2)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=100, verbose=False, min_lr=1e-5,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "valid/loss",
            },
        }

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self.mlp(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train/loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self.mlp(x)
        loss = F.mse_loss(y_hat, y)
        self.log('valid/loss', loss)