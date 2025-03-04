from typing import Any, Callable, Dict, IO, Iterable, List, Optional, Tuple, TypeVar, Union
import torch
import pytorch_lightning as pl
from torchvision.datasets.cifar import CIFAR10
import math
from datasets.utils.pad_tensors import PadTensors
from torch.utils.data import DataLoader
import numpy as np


class CIFAR10Wrapper(CIFAR10):
    """Sequential `CIFAR-10 <https://www.cs.toronto.edu/~kriz/cifar.html>`
    
    The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

    The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random order, but some training batches may contain more images from one class than another. Between them, the training batches contain exactly 5000 images from each class. 

    The 32x32 colour images are flattened. The first 1024 entries contain the red channel values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major order, so that the first 32 entries of the array are the red channel values of the first row of the image. 

    The labels range from 0-9.

    Args:
        save_to (str or ``pathlib.Path``): Root directory of dataset where directory
            ``cifar-10-batches-py`` will be saved to.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        ignore_first_timesteps (int, default: 900): ignore first n output timesteps for classification
        amplification (float, default: 1): amplify value range [0,1]
    """
    # code from https://pytorch.org/vision/master/_modules/torchvision/datasets/cifar.html#CIFAR10

    dataset_name = "SCIFAR-10"

    def __init__(
        self,
        save_to,
        train=True,
        ignore_first_timesteps: int = 700,
        amplification=1,  # scale input current to this maximum current value
    ):
        super().__init__(root=save_to, train=train, transform=None, target_transform=None, download=True)
        # flatten the images to (height*width, channels)
        self.data = self.data.reshape(self.data.shape[0], 1024, 3)

        self.ignore_first_timesteps = ignore_first_timesteps
        self.amplification = amplification

    def __getitem__(self, index):
        image = torch.from_numpy(self.data[index] / 255 * self.amplification).float()
        target = self.targets[index]

        target = np.int64(target)  # target must be numpy int
        block_idx = torch.ones((image.shape[0],), dtype=torch.int64)
        block_idx[:self.ignore_first_timesteps] = 0
        return image, target, block_idx

class CIFAR10LDM(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        batch_size: int = 32,
        num_workers: int = 1,
        name: str = None,  # for hydra
        num_classes: int = 10,  # for hydra
        valid_fraction: float = 0.05,
        random_seed = 42,
        amplification=1,
        ignore_first_timesteps: int = 700,
        input_size = 3  # for spike_plot.py
    ) -> None:
        super().__init__()
        self.data_path = data_path
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

        self.data_test = CIFAR10Wrapper(
            save_to=self.data_path,
            train=False,
            amplification=self.amplification,
            ignore_first_timesteps=self.ignore_first_timesteps
        )

        self.train_val_ds = CIFAR10Wrapper(
            save_to=self.data_path,
            train=True,
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
