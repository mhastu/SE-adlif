import pytorch_lightning as pl
import torch
from tonic.datasets.s_mnist import SMNIST
import math
from typing import Optional
from torch.utils.data import DataLoader
from datasets.utils.pad_tensors import PadTensors
import numpy as np


def current_encode(image, sensor_length = 1):
    return torch.from_numpy(image[:, None] / 255).float()  # the model wants input shape (time, x), where x is the flattened sensor size

def spike_encode(image, sensor_length):
    # Determine how many neurons should encode onset and offset
    half_size = sensor_length // 2

    # Determine thresholds of each neuron
    thresholds = np.linspace(0.0, 254.0, half_size).astype(np.uint8)

    # Determine for which pixels each neuron is above or below its threshol
    lower = image[:, None] < thresholds[None, :]
    higher = image[:, None] >= thresholds[None, :]

    # Get onsets and offset (transitions between lower and higher) spike times and ids
    on_spike_frame = np.logical_and(lower[:-1], higher[1:])
    off_spike_frame = np.logical_and(higher[:-1], lower[1:])

    # Get times when image is 255 and create matching neuron if
    touch_spike_frame = (image == 255)[1:, None]

    frames = np.concatenate((on_spike_frame, off_spike_frame, touch_spike_frame), axis=1)
    frames = torch.from_numpy(frames).float()
    return frames


# inherit from SMNIST for image downloading etc. but rewrite __getitem__ for current-encdoing with fixed number of frames (783 or 784)
# because SMNIST from tonic only outputs events and ToFrame would then cut off leading or trailing idleness which also contains information
class SMNISTWrapper(SMNIST):
    """Spiking/current-encoded sequential MNIST with fixed number of frames. Wrapper to support block_idx"""
    dataset_name = "SMNIST"

    def __init__(
        self,
        save_to,
        train=True,
        duplicate=False,
        num_neurons=33,
        ignore_first_timesteps: int = 700,
        amplification=1,  # scale input current to this maximum current value
    ):
        # transforms are None: the wrapper already outputs the correct datatype for the model, so it
        # interprets it as input current (float)
        # dt is not used, since we completely overwrite __getitem__() to use an implicit dt of 1ms
        super().__init__(save_to, train=train, duplicate=duplicate, num_neurons=num_neurons, transform=None, target_transform=None)
        self.amplification = amplification
        self.ignore_first_timesteps = ignore_first_timesteps

        self.encode = spike_encode
        if num_neurons == 1:
            self.encode = current_encode

    def __getitem__(self, index):
        frames = self.encode(self.image_data[index], self.sensor_size[0])
        frames = frames * self.amplification
        target = self.label_data[index]

        target = target.astype(np.int64)  # fix in pad_sequence "RuntimeError: value cannot be converted to type uint8_t without overflow"
        block_idx = torch.ones((frames.shape[0],), dtype=torch.int64)
        block_idx[:self.ignore_first_timesteps] = 0
        return frames, target, block_idx

class SMNISTLDM(pl.LightningDataModule):
    def __init__(
        self,
        data_path: str,
        input_size: int = 33,  # number of input neurons, must be odd
        batch_size: int = 32,
        num_workers: int = 1,
        name: str = None,  # for hydra
        num_classes: int = 10,  # for hydra
        valid_fraction: float = 0.05,
        random_seed = 42,
        amplification=1,
        ignore_first_timesteps: int = 10,
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

        self.input_size = input_size

        self.generator = torch.Generator().manual_seed(self.random_seed)

        self.data_test = SMNISTWrapper(
            save_to=self.data_path,
            train=False,
            num_neurons=self.input_size,
            amplification=self.amplification,
            ignore_first_timesteps=self.ignore_first_timesteps
        )

        self.train_val_ds = SMNISTWrapper(
            save_to=self.data_path,
            train=True,
            num_neurons=self.input_size,
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
