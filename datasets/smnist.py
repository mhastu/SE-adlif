import pytorch_lightning as pl
import torch
from tonic.datasets.s_mnist import SMNIST
import math
from typing import Optional
from torch.utils.data import DataLoader
from datasets.utils.pad_tensors import PadTensors
from tonic.transforms import ToFrame, Compose
from datasets.utils.transforms import Flatten
import numpy as np


class SMNISTWrapper(SMNIST):
    """Wrapper to support block_idx"""
    dataset_name = "SMNIST"
    def __getitem__(self, index):
        events, target = super().__getitem__(index)
        target = target.astype(np.int64)  # fix in pad_sequence "RuntimeError: value cannot be converted to type uint8_t without overflow"
        block_idx = torch.ones((events.shape[0],), dtype=torch.int64)
        return events, target, block_idx

class SMNISTLDM(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        window_size: float = 1000.0,  # should also be in us, since dt is in us
        input_size: int = 33,  # number of input neurons, must be odd
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
        self.window_size = window_size
        self.dt = dt
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_fraction = valid_fraction
        self.random_seed = random_seed

        # TODO: try without PadTensors() (then a default collate_fn is used)
        self.collate_fn = PadTensors()
        self.output_size = num_classes

        self.input_size = input_size
        sensor_size = (input_size, 1, 1)
        _event_to_tensor = ToFrame(
            sensor_size=sensor_size, time_window=self.window_size
        )
        def event_to_tensor(x):
            return torch.from_numpy(_event_to_tensor(x)).float()

        self.static_data_transform = Compose([
            event_to_tensor,
            Flatten()
        ])

        self.generator = torch.Generator().manual_seed(self.random_seed)

        self.data_test = SMNISTWrapper(
            save_to=self.data_path,
            train=False,
            num_neurons=num_neurons,
            dt=self.dt,
            transform=self.static_data_transform
        )

        self.train_val_ds = SMNISTWrapper(
            save_to=self.data_path,
            train=True,
            num_neurons=self.input_size,
            dt=self.dt,
            transform=self.static_data_transform
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
