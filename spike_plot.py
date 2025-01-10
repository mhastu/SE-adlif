import torch
import os
import hydra
from models.pl_module import MLPSNN
from omegaconf import OmegaConf
import math
import matplotlib.pyplot as plt
import numpy as np
import argparse
from mpl_toolkits.axes_grid1 import make_axes_locatable

parser = argparse.ArgumentParser(description="Plot spiking events or current values of all layers (input, 1, 2, output) when passing one sample through the model, given a checkpoint file. Model configuration is taken from parent directory of checkpoint path (hparams.yaml)")

parser.add_argument(
    "ckpt_filepath",
    type=str,
    help="Path to the checkpoint file."
)
parser.add_argument(
    "-d", "--device", 
    type=str, 
    default="cpu", 
    help="Device to use (e.g., 'cpu' or 'cuda:0'), default: 'cpu'."
)
parser.add_argument(
    "-i", "--i-in-batch", 
    type=int, 
    default=0, 
    help="Sample index in batch."
)

args = parser.parse_args()

ckpt_filepath = args.ckpt_filepath
device = args.device
i_in_batch = args.i_in_batch

cwd = os.path.dirname(os.path.dirname(ckpt_filepath))

time = os.path.basename(cwd)
date = os.path.basename(os.path.dirname(cwd))

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

# Load hyperparameter configuration
cfg_path = os.path.join(cwd, "logs/mlp_snn/version_0/hparams.yaml")
cfg = OmegaConf.load(cfg_path).cfg

cfg.input_size = cfg.dataset.input_size  # restore messed-up input_size (see pl_module.py)

# Load dataset
datamodule = hydra.utils.instantiate(cfg.dataset)
datamodule.setup("test")

# Load the model
model = MLPSNN_spiketrain.load_from_checkpoint(
    checkpoint_path=ckpt_filepath,
    cfg=cfg
)

# Set the model to evaluation mode
model.eval()

# Load a random sample from the dataset
test_loader = datamodule.test_dataloader()
data_iter = iter(test_loader)
inputs, targets, _ = next(data_iter)
print(inputs.shape)

# Move inputs to the configured device
if device is None:
    device = cfg.device
inputs = inputs.to(device)

# Forward pass to compute the spike train
with torch.no_grad():
    outseqs, l1seqs, l2seqs = model(inputs)


# ------------- PLOTS ----------------
input = inputs[i_in_batch]
outseq = outseqs[i_in_batch]
l1seq = l1seqs[i_in_batch]
l1seq = torch.where(l1seq != 0, torch.tensor(1, dtype=l1seq.dtype), torch.tensor(0, dtype=l1seq.dtype))  # map all non-zero values to 1
if l2seqs is not None:
    l2seq = l2seqs[i_in_batch]
    l2seq = torch.where(l2seq != 0, torch.tensor(1, dtype=l2seq.dtype), torch.tensor(0, dtype=l2seq.dtype))  # map all non-zero values to 1

num_subplots = 3 if l2seqs is None else 4
i_subplot = 0

xlim = [0, input.shape[0]]
maxaxheight = 6
minaxheight = 0.1

axheights = []
inputaxheight = input.shape[1]/100/xlim[1]*1000
if inputaxheight < minaxheight:
    inputaxheight = minaxheight
if inputaxheight > maxaxheight:
    inputaxheight = maxaxheight
axheights.append(inputaxheight)
l1axheight = l1seq.shape[1]/70/xlim[1]*1000
if l1axheight < minaxheight:
    l1axheight = minaxheight
if l1axheight > maxaxheight:
    l1axheight = maxaxheight
axheights.append(l1axheight)
if l2seqs is not None:
    l2axheight = l2seq.shape[1]/70/xlim[1]*1000
    if l2axheight < minaxheight:
        l2axheight = minaxheight
    if l2axheight > maxaxheight:
        l2axheight = maxaxheight
    axheights.append(l2axheight)
outputaxheight = outseq.shape[1]/20/xlim[1]*1000
if outputaxheight < minaxheight:
    outputaxheight = minaxheight
if outputaxheight > maxaxheight:
    outputaxheight = maxaxheight
axheights.append(outputaxheight)

# heatmap for the input values
fig, axs = plt.subplots(num_subplots,1,figsize=(12,sum(axheights)), gridspec_kw={'height_ratios': axheights})

if input.shape[1] == 1:
    axs[i_subplot].get_yaxis().set_ticks([])  # no tick labels for only 1 input neuron
axs[i_subplot].set_ylabel("Input")
im = axs[i_subplot].imshow(input.T, aspect="auto", cmap="viridis", origin="lower", interpolation="nearest")
divider = make_axes_locatable(axs[i_subplot])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='horizontal' if input.shape[1] == 1 else 'vertical')
i_subplot += 1

# event plot for the spike train
axs[i_subplot].set_ylabel(f"Layer 1 ({l1seqs[i_in_batch].mean()*100:.2f}%)")
axs[i_subplot].set_xlim(xlim)
axs[i_subplot].imshow(l1seq.T, aspect="auto", cmap="Greys", origin="lower", interpolation="nearest")
divider = make_axes_locatable(axs[i_subplot])
cax = divider.append_axes('right', size='5%', pad=0.05)
cax.set_visible(False)
i_subplot += 1

if l2seqs is not None:
    # event plot for the spike train
    axs[i_subplot].set_ylabel(f"Layer 2 ({l2seqs[i_in_batch].mean()*100:.2f}%)")
    axs[i_subplot].set_xlim(xlim)
    axs[i_subplot].imshow(l2seq.T, aspect="auto", cmap="Greys", origin="lower", interpolation="nearest")
    divider = make_axes_locatable(axs[i_subplot])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cax.set_visible(False)
    i_subplot += 1

# output heatmap
axs[i_subplot].set_xlabel("Time Steps")
axs[i_subplot].set_ylabel("Output")
im = axs[i_subplot].imshow(outseq.T, aspect="auto", cmap="viridis", origin="lower", interpolation="nearest")
divider = make_axes_locatable(axs[i_subplot])
cax = divider.append_axes('right', size='5%', pad=0.05)
fig.colorbar(im, cax=cax, orientation='vertical')

# add target to filename
strtarget = ""
target = int(targets[i_in_batch, 1])
strtarget = "_" + str(target)[:10]

fig.tight_layout()
fig.savefig(f"spikeplot_{date}_{time}{strtarget}.pdf")
