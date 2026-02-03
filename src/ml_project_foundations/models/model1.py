"""
Example Model 2 definition using PyTorch Lightning.
"""

import lightning.pytorch as pl
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch import nn
from torch.optim import Adam


class Model1(pl.LightningModule):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters(
            OmegaConf.to_container(model_cfg, resolve=True)
        )  # Saves hyperparameters to self.hparams
        self.layer1 = nn.Linear(model_cfg.input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 1)
        self.cfg = model_cfg

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.output_layer(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log("test_loss", loss)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams.lr)
