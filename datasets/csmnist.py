import pytorch_lightning as pl
import torch
from tonic.datasets.s_mnist import SMNIST
import math
from typing import Optional
from torch.utils.data import DataLoader
from datasets.utils.pad_tensors import PadTensors
import numpy as np


# inherit from CSMNIST from SMNIST for image downloading etc. but rewrite __getitem__ for current-encoding instead of spike events
class CSMNISTWrapper(SMNIST):
    """Current-encoded sequential MNIST with 1 input neuron. Wrapper to support block_idx"""
    dataset_name = "CSMNIST"

    def __init__(
        self,
        save_to,
        train=True,
        duplicate=True,
        dt=1000.0,
        amplification=1,  # scale image data to this maximum current value
        ignore_first_timesteps: int = 700,
    ):
        # transforms are None: the wrapper already outputs the correct datatype for the model, so it
        # interprets it as input current (float)
        super().__init__(save_to, train=train, duplicate=duplicate, num_neurons=1, dt=dt, transform=None, target_transform=None)
        self.amplification = amplification
        self.ignore_first_timesteps = ignore_first_timesteps

    def __getitem__(self, index):
        image = self.image_data[index] / 255 * self.amplification
        image = torch.from_numpy(image[:, None]).float()  # the model wants input shape (time, x), where x is the flattened sensor size
        target = self.label_data[index]

        target = target.astype(np.int64)  # fix in pad_sequence "RuntimeError: value cannot be converted to type uint8_t without overflow"
        block_idx = torch.ones((image.shape[0],), dtype=torch.int64)
        block_idx[:self.ignore_first_timesteps] = 0
        return image, target, block_idx

class CSMNISTLDM(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        window_size: float = 1000.0,  # should also be in us, since dt is in us
        dt: float = 1000,  # duration (in us) for each timestep
        batch_size: int = 32,
        num_workers: int = 1,
        name: str = None,  # for hydra
        num_classes: int = 10,  # for hydra
        valid_fraction: float = 0.05,
        random_seed = 42,
        amplification=1,
        ignore_first_timesteps: int = 10
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.window_size = window_size
        self.dt = dt
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_fraction = valid_fraction
        self.random_seed = random_seed
        self.amplification = amplification
        self.ignore_first_timesteps = ignore_first_timesteps

        # TODO: try without PadTensors() (then a default collate_fn is used)
        self.collate_fn = PadTensors()
        self.output_size = num_classes

        self.generator = torch.Generator().manual_seed(self.random_seed)

        self.data_test = CSMNISTWrapper(
            save_to=self.data_path,
            train=False,
            dt=self.dt,
            amplification=self.amplification,
            ignore_first_timesteps=self.ignore_first_timesteps
        )

        self.train_val_ds = CSMNISTWrapper(
            save_to=self.data_path,
            train=True,
            dt=self.dt,
            amplification=self.amplification,
            ignore_first_timesteps=self.ignore_first_timesteps
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
