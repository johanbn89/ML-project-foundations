""" """

import warnings

import lightning as pl
import pandas as pd
import torch
from data_quarry.tools import get_file_paths
from omegaconf import DictConfig, ListConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import Compose

warnings.filterwarnings("ignore")


# Dummy dataset class for illustration
class DummyDataset(Dataset):
    def __init__(self, data: dict, cfg: DictConfig | ListConfig, transform: list):
        self.cfg = cfg
        self.input = pd.read_csv(data[self.cfg.components[0]][0])
        self.target = pd.read_csv(data[self.cfg.components[1]][0])
        self.transform = Compose(transform)

        # print("Shapes:", self.input.shape, self.target.shape)

    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        input = torch.tensor(self.input.iloc[idx].values, dtype=torch.float32)
        target = torch.tensor(self.target.iloc[idx].values, dtype=torch.float32)
        if self.transform:
            input = self.transform(input)
        return input, target


class DataModule(pl.LightningDataModule):
    def __init__(self, data_cfg: DictConfig | ListConfig):
        super().__init__()
        self.save_hyperparameters(data_cfg)  # Saves to self.hparams
        self.cfg = data_cfg

    def prepare_data(self):
        # Download or prepare data if needed
        # This happens only once on one GPU/TPU in distributed settings
        # ACTUALLY it should return a dataset object?
        data = get_file_paths(
            dataset=self.cfg.name,
            ref=self.cfg.tag,
            components=self.cfg.components,
        )
        self.data = DummyDataset(data, self.cfg, self.cfg.transforms)

    def setup(self, stage=None):
        # Load and split the dataset
        # TODO: Make use of self.cfg to customize dataset loading
        full_dataset = self.data
        train_size = int(0.8 * len(full_dataset))
        val_size = int(0.1 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
        )


if __name__ == "__main__":
    cfg = OmegaConf.load("src/ml_project_foundations/config/data/base.yaml")
    print(OmegaConf.to_yaml(cfg))

    print(cfg.name, cfg.tag, cfg.components)
    data = get_file_paths(dataset=cfg.name, ref=cfg.tag, components=cfg.components)
    print(data)

    data_module = DataModule(cfg)
    data_module.prepare_data()
    data_module.setup()
    train_loader = data_module.train_dataloader()
    for batch in train_loader:
        # LOOKS GOOD
        print(batch)
        print(type(batch))
        print(len(batch))
        print(batch[0].shape)
        break
