import warnings

warnings.filterwarnings("ignore", category=UserWarning)
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import lightning as pl
import pandas as pd
import torch
from data_quarry.tools import get_file_paths
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.transforms import Compose


# Dummy dataset class for illustration
class DummyDataset(Dataset):
    def __init__(
        self,
        data: Mapping[str, Sequence[Path]],
        components: Sequence[str],
        transforms: Sequence[Callable[[torch.Tensor], torch.Tensor]] | None,
    ) -> None:
        if len(components) < 2:
            raise ValueError("DummyDataset requires input and target components.")
        self.input = pd.read_csv(data[components[0]][0])
        self.target = pd.read_csv(data[components[1]][0])
        self.transform = Compose(list(transforms)) if transforms else None

    def __len__(self) -> int:
        return len(self.input)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        input_tensor = torch.tensor(self.input.iloc[idx].values, dtype=torch.float32)
        target = torch.tensor(self.target.iloc[idx].values, dtype=torch.float32)
        if self.transform is not None:
            input_tensor = self.transform(input_tensor)
        return input_tensor, target


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        pin_memory: bool,
        shuffle: bool,
        drop_last: bool,
        name: str,
        tag: str,
        components: Sequence[str],
        transforms: Sequence[Callable[[torch.Tensor], torch.Tensor]] | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore="transforms")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.name = name
        self.tag = tag
        self.components = list(components)
        self.transforms = list(transforms) if transforms else None

    def prepare_data(self) -> None:
        # Download or prepare data if needed
        # This happens only once on one GPU/TPU in distributed settings
        # ACTUALLY it should return a dataset object?
        data = get_file_paths(
            dataset=self.name,
            ref=self.tag,
            components=self.components,
        )
        self.data = DummyDataset(data, self.components, self.transforms)

    def setup(self, stage: str | None = None) -> None:
        # Load and split the dataset
        full_dataset = self.data
        train_size = int(0.8 * len(full_dataset))
        val_size = int(0.1 * len(full_dataset))
        test_size = len(full_dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            full_dataset, [train_size, val_size, test_size]
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )


if __name__ == "__main__":
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    cfg = OmegaConf.load("src/ml_project_foundations/config/data/base.yaml")
    print(OmegaConf.to_yaml(cfg))

    data_module = instantiate(cfg)
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
