import torch
import os
import hydra
from models.pl_module import MLPSNN  # Import your SNN model
from omegaconf import OmegaConf
import math
import matplotlib.pyplot as plt
import numpy as np

# TODO: argparse --------------------
ckpt_path = "results/hydra/2024-11-30/13-23-55/ckpt/epoch=19-step=2220.ckpt"
device = "cpu"  # type: str | None
i_in_batch = 0
# -----------------------------------

cwd = os.path.dirname(os.path.dirname(ckpt_path))

class MLPSNN_spiketrain(MLPSNN):
    def forward(
        self, inputs: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        s1 = self.l1.initial_state(inputs.shape[0], inputs.device)
        s_out = self.out_layer.initial_state(inputs.shape[0], inputs.device)
        if self.two_layers:
            s2 = self.l2.initial_state(inputs.shape[0], inputs.device)
        out_sequence = []
        l1_sequence = []
        l2_sequence = []
        single_step_prediction_limit = int(math.ceil(inputs.shape[1] * 0.5))

        # Iterate over each time step in the data
        for t, x_t in enumerate(inputs.unbind(1)):

            # Auto-regression for oscillator task
            if self.auto_regression and t >= single_step_prediction_limit:
                x_t = out.detach()
            out, s1 = self.l1(x_t, s1)
            l1_sequence.append(out)
            out = torch.nn.functional.dropout(out, p=self.dropout, training=self.training)
            if self.two_layers:
                out, s2 = self.l2(out, s2)
                l2_sequence.append(out)
                out = torch.nn.functional.dropout(out, p=self.dropout, training=self.training)
            out, s_out = self.out_layer(out, s_out)
            # out[:,0] += 100
            out_sequence.append(out)
        
        l2_sequence_ = torch.stack(l2_sequence, dim=1) if self.two_layers else None
        return torch.stack(out_sequence, dim=1), torch.stack(l1_sequence, dim=1), l2_sequence_

# Load configuration
cfg_path = os.path.join(cwd, "logs/mlp_snn/version_0/hparams.yaml")
cfg = OmegaConf.load(cfg_path).cfg

cfg.input_size = cfg.input_layer_size  # restore messed-up input_size (see pl_module.py)

# Load dataset
datamodule = hydra.utils.instantiate(cfg.dataset)
datamodule.setup("test")  # Ensure the dataloader is prepared

# Load the model
model = MLPSNN_spiketrain.load_from_checkpoint(
    checkpoint_path=ckpt_path,
    cfg=cfg
)

# Set the model to evaluation mode
model.eval()

# Load a random sample from the dataset
test_loader = datamodule.test_dataloader()
data_iter = iter(test_loader)
inputs, targets, _ = next(data_iter)  # Assuming the dataset provides inputs, targets, and block_idx

# Move inputs to the configured device
if device is None:
    device = cfg.device
inputs = inputs.to(device)

# Forward pass to compute the spike train
with torch.no_grad():
    outseq, l1seq, l2seq = model(inputs)


# ------------- PLOTS ----------------
inputs = inputs[i_in_batch]
outseq = outseq[i_in_batch]
l1seq = l1seq[i_in_batch]
if l2seq is not None:
    l2seq = l2seq[i_in_batch]

plt.figure(figsize=(6,6))
plt.imshow(inputs.view(28, 28), aspect="auto", cmap="viridis", interpolation="nearest")
plt.savefig("input.png")
plt.close()

num_subplots = 3 if l2seq is None else 4
i_subplot = 1

xlim = [0, inputs.shape[0]]

# heatmap for the input values
plt.figure(figsize=(12, 6))
plt.subplot(num_subplots, 1, i_subplot)
plt.ylabel("Input")
plt.imshow(inputs.T, aspect="auto", cmap="viridis", origin="lower", interpolation="nearest")
i_subplot += 1

# event plot for the spike train
plt.subplot(num_subplots, 1, i_subplot)
plt.ylabel("Layer 1")
# Convert the spike train into an event list format
neuron_indices, time_steps = np.where(l1seq.T == 1)  # Find where spikes occur
events = [[] for _ in range(l1seq.shape[1])]  # Create a list for each neuron
for neuron, time in zip(neuron_indices, time_steps):
    events[neuron].append(time)
plt.xlim(xlim)
plt.eventplot(events, orientation="horizontal", linelengths=0.8, colors="black")
plt.tight_layout()
i_subplot += 1

if l2seq is not None:
    # event plot for the spike train
    plt.subplot(num_subplots, 1, i_subplot)  # Bottom subplot for the spike train
    plt.ylabel("Layer 2")
    # Convert the spike train into an event list format
    neuron_indices, time_steps = np.where(l2seq.T == 1)  # Find where spikes occur
    events = [[] for _ in range(l2seq.shape[1])]  # Create a list for each neuron
    for neuron, time in zip(neuron_indices, time_steps):
        events[neuron].append(time)
    plt.xlim(xlim)
    plt.eventplot(events, orientation="horizontal", linelengths=0.8, colors="black")
    plt.tight_layout()
    i_subplot += 1

# Create an event plot for the spike train
plt.subplot(num_subplots, 1, i_subplot)  # Bottom subplot for the spike train
plt.xlabel("Time Steps")
plt.ylabel("Output")
plt.imshow(outseq.T, aspect="auto", cmap="viridis", origin="lower", interpolation="nearest")
plt.tight_layout()

# Show the plots
plt.savefig("spikeplot.png")
