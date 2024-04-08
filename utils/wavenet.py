import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from utils import DEVICE

### Compression/decompression
# This requires some explanation. Audio is frequently digitized as an int16 (x)
# per sample, and represented as the float32 x/65536, which is between -1.0 and 1.0.
# These functions (de)compress x to a uint8 (integer between 0 and 255) in a way that
# supposedly retains audio quality.
#
# Why? When training a WaveNet for the generative task of predicting the next sample,
# the WaveNet authors wanted to treat this as a classification problem. Companding
# reduces the number of classes from 65536 to 256.
#
# If we want to train on the generative task for EEGs, we need to think about what
# should replace this. One option would be to forget about compression entirely and
# just use a regression loss, e.g. mean-squared-error loss. This will require modifying
# the generative head to output samples rather than logits.

def compand(x):
    """Mu-law companding transformation, which compresses a 16-bit depth
    signal to 8 bits. We use uint8 to make it easier to treat the outputs
    as classification labels.
    
    Args:
      x: torch.Tensor with values between -1 and 1. Generally speaking,
        this should be float32 obtained by scaling an int16 signal down
        to [-1, 1].
    Returns:
      torch.Tensor of the same shape as x, with torch.uint8 dtype
    """
    mu = 255
    out = torch.sign(x) * torch.log(1 + mu*torch.abs(x)) / np.log(1 + mu)
    return (128*out + 128).to(torch.uint8)

def inverse_compand(y):
    """Inverse of mu-law companding transformation, expanding an 8-bit integer
    signal to bit depth 16.
    
    Args:
      x: torch.Tensor with torch.uint8 dtype.
    Returns:
      torch.Tensor of the same shape as x, with torch.float dtype, lying in
        [-1, 1].
    """
    mu = 255
    z = y.float() / 128 - 1
    return torch.sign(z) * ((1 + mu)**torch.abs(z) - torch.ones_like(z)) / mu


### Data loading
# This function takes a batch of shape [batch, channels, seq+1] and produces a tensor
# and targets of shape [batch, channels, seq], suitable for the task of predicting
# the next sample.

# def prep_batch_generative(batch):
#     """Given a batch, get a pair (batch, targets). The targets are the batch,
#     shifted left by 1, and converted to one of 256 classes per sample by
#     companding. 
    
#     Note that one sample gets dropped: a dataloader producing a sequence length
#     of 9 is needed for a model that takes sequences of length 8.
#     """
#     tens = batch[:, :, :-1]  # batch, channels, seq
#     targ = compand(batch[:, :, 1:])
#     return tens, targ

def prep_batch_generative(batch):
    # EEG signals have large amplitudes -- scaling them down so the
    # generative loss doesn't overwhelm the discriminative one
    return batch[:, :, :-1], batch[:, :, 1:] / 65536


### Dilation
# WaveNet achieves large context windows by using increasing dilations as one moves
# through the convolutional layers. Here are two alternative approaches; feel free
# to try your own!

def default_dilation_schedule():
    """Generator for the dilation pattern used in the WaveNet paper."""
    dilation = 1
    while True:
        yield dilation
        dilation = 2 * dilation
        if dilation > 512:
            dilation = 1
            
def doubling_dilation_schedule():
    """Continue to double dilations forever."""
    dilation = 1
    while True:
        yield dilation
        dilation = 2 * dilation


### Architecture

class CausalConv1d(nn.Module):
    """Causal 1D convolution. Outputs are the same shape as the inputs,
    and right-aligned so that the output at index n only depends on inputs
    at indices before n.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1,
                 bias=False, device=None, dtype=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=1,
                              padding=0, dilation=dilation, bias=bias,
                              device=device, dtype=dtype)
        if device is not None:
            self.to(device)
        
    def forward(self, x):
        # Left pad x
        x = F.pad(x, (self.dilation*(self.kernel_size-1), 0), mode="constant", value=0)
        # Now output is the same shape as x, and right-aligned
        return self.conv(x)
    
class SigmoidTanh(nn.Module):
    """Activation function consisting of a sigmoid on half of the channels,
    multiplied by a tanh on the other half.
    """
    def __init__(self):
        super().__init__()
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        num_channels = x.shape[1]
        assert num_channels % 2 == 0
        return self.tanh(x[:, :num_channels//2]) * self.sigmoid(x[:, num_channels//2:])
    
class ResidualBlock(nn.Module):
    """Block featuring a causal convolution layer, a sigmoid-tanh activation,
    and a 1x1 convolution that allows local mixing of features between channels.
    Produces two outputs, a "residual" and "skip" connection, with the skip
    connection intended to be sent to the output layer, and the residual
    connection intended to be sent to the next block.
    """
    def __init__(self, channels, kernel_size, dilation=1, bias=False,
                 device=None, dtype=None):
        super().__init__()
        self.conv = CausalConv1d(channels, 2*channels, kernel_size,
                                 dilation=dilation, bias=bias, device=device,
                                 dtype=dtype)
        self.actn = SigmoidTanh()
        self.onexone = nn.Conv1d(channels, channels, kernel_size=1, bias=bias,
                                 device=device, dtype=dtype)
        if device is not None:
            self.to(device)
            
    def forward(self, x):
        # Skip connection sent to final output
        # Res connection sent to next layer
        skip = self.onexone(self.actn(self.conv(x)))
        res = x + skip
        return res, skip
    
        
class ResidualStack(nn.Module):
    """A sequence of ResidualBlocks with different dilations.
    
    dilation_schedule should be an iterable yielding a sequence of dilations.
    By default, the schedule in the paper is used:
    1, 2, 4, ..., 512, 1, 2, 4, ..., 512, 1, 2, ...
    """
    def __init__(self, channels, num_blocks, kernel_size, 
                 dilation_schedule=None, bias=False, device=None, dtype=None):
        super().__init__()
        if dilation_schedule is None:
            dilation_schedule = default_dilation_schedule()
        self.blocks = []
        self.receptive_field = 1
        for dil in dilation_schedule:
            self.blocks.append(ResidualBlock(channels, kernel_size, dil, bias,
                                             device=device, dtype=dtype))
            self.receptive_field += dil*(kernel_size-1)
            if len(self.blocks) >= num_blocks:
                break
        self.blocks = nn.ModuleList(self.blocks)
        if device is not None:
            self.to(device)
        
    def forward(self, x):
        skip_out = torch.zeros_like(x)
        for block in self.blocks:
            x, skip = block(x)
            skip_out = skip_out + skip
        return skip_out
    
    
# This needs to be modified if we want to try generative training WITHOUT mu-law companding.
# One option would be to have num_classes = in_channels (so it's outputting raw signals),
# and use a regression loss.
# Another would be to output a 4d tensor [batch, out_channels, seq, classes], and
# use a classification loss, against some compressed form of the original signal.

# Because of time constraints, I ended up keeping the architecture the same, and just
# using a regression loss -- so the outputs are interpreted as [batch, channels, seq],
# not [batch, channels, classification_logits].

class WaveNetGenerativeHead(nn.Module):
    """Receive the sum of skip connections from the ResBlock layers.
    Produce logits for next audio sample, taken as one of 256 companded
    classes.
    
    For a generative model, the output will have the same sequence length as
    the input. We can view it as a sequence of predictions, one for each
    timestep, which is then evaluated against the inputs shifted 1 timestep
    backwards.
    """
    def __init__(self, channels, num_classes=256):
        super().__init__()
        self.steps = nn.Sequential(
            nn.ReLU(),
            # Downsample to 256 classes
            nn.Conv1d(channels, num_classes, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(num_classes, num_classes, kernel_size=1),
            # Outputs treated as logits
        )
        
    def forward(self, x):
        return self.steps(x)
    
    
class WaveNetGenerative(nn.Module):
    """WaveNet architecture for generating raw audio."""
    def __init__(self, in_channels, hidden_channels, num_blocks, kernel_size,
                 dilation_schedule=None, bias=False, device=None, dtype=None):
        super().__init__()
        self.input = CausalConv1d(in_channels, hidden_channels, kernel_size, bias=bias,
                                  device=device, dtype=dtype)
        self.res_stack = ResidualStack(hidden_channels, num_blocks, kernel_size,
                                       dilation_schedule, bias=bias,
                                       device=device, dtype=dtype)
        self.output = WaveNetGenerativeHead(hidden_channels)
        
        self.receptive_field = self.res_stack.receptive_field
        if device is None:
            device = torch.device(DEVICE)
        self.device = device
        self.to(device)
        
    def forward(self, x):
        # x in shape [batch, channels, seq]
        # Return in shape [batch, channels, seq]
        return self.output(self.res_stack(self.input(x)))
    
    def generate(self, start_audio, num_new_samples):
        # Assume start_audio is a batch
        out_audio = torch.cat([
            start_audio,
            torch.zeros(start_audio.shape[0], num_new_samples, device=self.device)
        ], dim=1)
        for i in range(num_new_samples):
            # Restrict to samples visible by the model
            start_idx = max(0, start_audio.shape[1] + i - self.receptive_field)
            audio = out_audio[:, start_idx:start_audio.shape[1]+i]
            next_sample_logits = self(audio)[:, -1, :]
            next_sample_prob = F.softmax(next_sample_logits, dim=1)
            next_sample = inverse_compand(torch.multinomial(next_sample_prob, 1))
            out_audio[:, start_audio.shape[1]+i] = next_sample.squeeze()
        return out_audio
    

class WaveNetClassifierHead(nn.Module):
    def __init__(self, channels, downsample, num_classes):
        super().__init__()
        self.pre_pool = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=5, padding=2),
            nn.ReLU(),
        )
        self.pool = nn.AvgPool1d(downsample, stride=downsample)
        # shape [batch, channels, seq // downsample]
        self.post_pool = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(channels, num_classes, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Conv1d(num_classes, num_classes, kernel_size=5, padding=2),
        )
        # shape [batch, num_classes, seq // downsample]
        
    def forward(self, x):
        x = self.post_pool(self.pool(self.pre_pool(x)))
        x = x.mean(dim=2)  # shape [batch, num_classes]; average over time
        x = F.log_softmax(x, dim=1)  # convert to log-probabilities
        return x
    
    
class WaveNetDiscriminative(nn.Module):
    """WaveNet architecture for classifying audio."""
    def __init__(self, in_channels, hidden_channels, num_blocks, kernel_size, downsample, 
                 num_classes, dilation_schedule=None, bias=False, device=None,
                 dtype=None):
        super().__init__()
        self.input = CausalConv1d(in_channels, hidden_channels, kernel_size, bias=bias,
                                  device=device, dtype=dtype)
        self.res_stack = ResidualStack(hidden_channels, num_blocks, kernel_size,
                                       dilation_schedule, bias=bias,
                                       device=device, dtype=dtype)
        self.output = WaveNetClassifierHead(hidden_channels, downsample, num_classes)
        
        self.receptive_field = self.res_stack.receptive_field
        if device is None:
            device = torch.device(DEVICE)
        self.device = device
        self.to(device)
        
    def forward(self, x):
        # x in shape [batch, channels, seq]
        # Return in shape [batch, logit]
        return self.output(self.res_stack(self.input(x)))
    
    
class WaveNetDualTask(nn.Module):
    """WaveNet producing both generative and discriminative outputs. (For us,
    "discriminative" means producing log-probabilities which are fed into a
    KL-divergence loss.)
    
    Params:
      in_channels: Number of input channels (e.g. 20 to use all EEG channels).
      hidden_channels: Number of channels used internally by convolution layers.
      num_blocks: Number of successive convolution blocks.
      kernel_size: Size of convolution kernel.
      gen_channels_out: Number of channels for the generative head to produce.
        For a regression loss, this should equal in_channels.
      downsample: How far to downsample before applying classifier head. If
        seq_length // downsample > 1, you'll get multiple classifier outputs per
        input signal, which you can then decide how to collate into a prediction.
      num_classes: Number of classes for classification (KL-divergence) task.
      dilation_schedule: Iterable giving the dilation of each convolution layer.
        The default starts at 1, doubles every layer up to 512, and then repeats.
      bias: Whether to include biases in convolution layers. Default: False.
      device: Device to store model on. Default: whatever's in the config file.
      dtype: Data type of inputs.
      
    WaveNetDualTask.receptive_field shows how many consecutive EEG samples are used
    to calculate a given output.
    """
    def __init__(self, in_channels, hidden_channels, num_blocks, kernel_size, gen_channels_out,
                 downsample, num_classes, dilation_schedule=None, bias=False, device=None,
                 dtype=None):
        super().__init__()
        self.input = CausalConv1d(in_channels, hidden_channels, kernel_size, bias=bias,
                                  device=device, dtype=dtype)
        self.res_stack = ResidualStack(hidden_channels, num_blocks, kernel_size,
                                       dilation_schedule, bias=bias,
                                       device=device, dtype=dtype)
        self.gen_output = WaveNetGenerativeHead(hidden_channels, gen_channels_out)
        self.disc_output = WaveNetClassifierHead(hidden_channels, downsample, num_classes)
        
        self.receptive_field = self.res_stack.receptive_field
        if device is None:
            device = torch.device(DEVICE)
        self.device = device
        self.to(device)
        
    def forward(self, x):
        # x in shape [batch, channels, seq]
        res_out = self.res_stack(self.input(x))
        # discriminative output in shape [batch, channels, seq]
        disc_out = self.disc_output(res_out)
        # generative output in shape [batch, channels, seq]
        gen_out = self.gen_output(res_out)
        return gen_out, disc_out
    
    def generate(self, start_audio, num_new_samples):
        # Assume start_audio is a batch
        out_audio = torch.cat([
            start_audio,
            torch.zeros(start_audio.shape[0], num_new_samples, device=self.device)
        ], dim=1)
        for i in range(num_new_samples):
            # Restrict to samples visible by the model
            start_idx = max(0, start_audio.shape[1] + i - self.receptive_field)
            audio = out_audio[:, start_idx:start_audio.shape[1]+i]
            next_sample_logits = self(audio)[0][:, -1, :]
            next_sample_prob = F.softmax(next_sample_logits, dim=1)
            next_sample = inverse_compand(torch.multinomial(next_sample_prob, 1))
            out_audio[:, start_audio.shape[1]+i] = next_sample.squeeze()
        return out_audio
