import pytorch_lightning as pl
import torch
from tonic.datasets.s_mnist import SMNIST
import math
from typing import Optional
from torch.utils.data import DataLoader
from datasets.utils.pad_tensors import PadTensors


class SMNISTWrapper(SMNIST):
    """Wrapper to support block_idx (only needed for compatibility with models,
    even though MNIST does not need padding)"""
    dataset_name = "SMNIST"
    def __getitem__(self, index):
        events, target = super().__getitem__(index)
        block_idx = torch.ones((events.shape[0],), dtype=torch.int64)
        return events, target, block_idx

def collate(batch):        
    inputs, target_list, block_idx = list(zip(*batch))  # type: ignore

    # If target is a scalar, convert it to a tensor
    if len(target_list[0].shape) == 0:
        target_list = torch.tensor(target_list).unsqueeze(1)

    return inputs, target_list, block_idx

class SMNISTLDM(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        input_size: int = 99,  # number of input neurons, must be odd
        dt: float = 1000,  # duration (in us) for each timestep
        batch_size: int = 32,
        num_workers: int = 1,
        name: str = None,  # for hydra
        num_classes: int = 10,  # for hydra
        valid_fraction: float = 0.05,
        random_seed = 42
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.dt = dt
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_fraction = valid_fraction
        self.random_seed = random_seed

        #self.collate_fn = PadTensors()
        self.collate_fn = collate
        self.input_size = input_size
        self.output_size = num_classes

        self.generator = torch.Generator().manual_seed(self.random_seed)

        self.data_test = SMNISTWrapper(
            save_to=self.data_path,
            train=False,
            num_neurons=self.input_size,
            dt=self.dt
        )

        self.train_val_ds = SMNISTWrapper(
            save_to=self.data_path,
            train=True,
            num_neurons=self.input_size,
            dt=self.dt
        )
        valid_len = math.floor(len(self.train_val_ds) * self.valid_fraction)
        self.data_train, self.data_val = torch.utils.data.random_split(
            self.train_val_ds,
            [len(self.train_val_ds) - valid_len, valid_len],
            generator=self.generator,
        )

    def prepare_data(self):
        pass
        

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            shuffle=True,
            pin_memory=True,
            batch_size=self.batch_size,
            drop_last=True,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            shuffle=False,
            pin_memory=True,
            batch_size=self.batch_size,
            drop_last=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            pin_memory=True,
            shuffle=False,
            batch_size=self.batch_size,
            drop_last=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.data_test,
            shuffle=False,
            batch_size=self.batch_size,
            pin_memory=True,
            drop_last=False,
            collate_fn=self.collate_fn,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )