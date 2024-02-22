### This file should not be edited directly. Run kaggle_sync.py to rebuild it.
### From kaggle_platform.py ###

### ~~~
## ~~~ Function that tests whether or not code is being executed in kaggle
### ~~~

import os

def this_is_running_in_kaggle():
    try:
        import kaggle
        import kaggle_secrets
        kaggle_packages_found = True
    except ImportError:
        kaggle_packages_found = False
    return ('kaggle' in os.getcwd() and kaggle_packages_found)

# written by Tom



### From config.py ###
from pathlib import Path
import torch


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LOG_DIR = Path("/home/pvk/Documents/python/ML/brains/logs/")
MODEL_SAVE_DIR = Path("/home/pvk/Documents/python/ML/brains/models/")
BASE_PATH = Path("data/")
SPEC_DIR = Path("data/spectrograms_npy")

if this_is_running_in_kaggle():
    BASE_PATH = Path("/kaggle/input/hms-harmful-brain-activity-classification")
    SPEC_DIR = Path("/tmp/dataset/hms-hbac")
    LOG_DIR = Path("/tmp/logs")
    MODEL_SAVE_DIR = Path("/tmp/models")


### From training.py ###
from pathlib import Path
from typing import Optional, List, Tuple, Callable
import gc
import time

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter

from tqdm.auto import tqdm



def cycle(iterable):
    # from https://github.com/pytorch/pytorch/issues/23900#issuecomment-518858050
    iterator = iter(iterable)
    while True:
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(iterable)

class Trainer():
    """Bundles together a model, a training and optionally a validation dataset,
    an optimizer, and loss/accuracy/fbeta@0.5 metrics. Stores histories of the
    metrics for visualization or comparison. Optionally writes metric hsitories
    to TensorBoard.
    """
    def __init__(self,
                 model: nn.Module,
                 train_loader: data.DataLoader,
                 val_loader: data.DataLoader,
                 *,
                 optimizer: type[torch.optim.Optimizer] = optim.SGD,
                 criterion: nn.Module = nn.KLDivLoss(reduction="mean"),
                 lr: float = 0.03,
                 scheduler: Optional[type[torch.optim.lr_scheduler]] = None,
                 writer: SummaryWriter | str | None = None,
                 model_name: str | None = None,
                 **kwargs,
                 ):
        """Create a Trainer.
        
        Args:
          model: Model to train.
          train_loader: DataLoader containing training data.
          val_loader: DataLoader containing validation data.
          optimizer: Optimizer to use during training; give it a class, not an
            instance (SGD, not SGD()). Default torch.optim.SGD.
          criterion: Loss criterion to minimize. Default nn.KLDivLoss().
          lr: Learning rate. Default 0.03.
          scheduler: Optional learning rate scheduler. As with optimizer, give
            a class, which will be initialized at the start of training. If no
            scheduler is given, a constant learning rate is used.
          writer: SummaryWriter object to log performance to TensorBoard. You
            can create this using .logging.create_writer(). If writer is
            "auto", a SummaryWriter will automatically be created, using
            model_name (which is required in this case).
          model_name: Name of the model. Will be used to save checkpoints for
            the model and to automatically create a SummaryWriter.
          Keyword arguments prefixed by `optimizer_` or `scheduler_` are passed
            to the optimizer or scheduler, respectively.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.criterion = criterion
        self.lr = lr
        self.scheduler_class = scheduler
        self.optimizer_kwargs = dict()
        self.scheduler_kwargs = dict()
        self.scheduler_kwargs.setdefault('max_lr', self.lr)
        for key in kwargs.keys():
            # strip off optimizer or scheduler prefix and add to relevant dict
            if key.startswith('optimizer_'):
                self.optimizer_kwargs.update({key[10:]: kwargs[key]})
            if key.startswith('scheduler_'):
                self.scheduler_kwargs.update({key[10:]: kwargs[key]})
        self.optimizer = optimizer(model.parameters(), lr=self.lr,
                                   **self.optimizer_kwargs)
        if self.scheduler_class is not None:
            self.scheduler = self.scheduler_class(
                self.optimizer,
                **self.scheduler_kwargs
            )
            
        self.histories = {
            'batches': [],
            'train_loss': [],
            'val_loss': [],
            'lr': []
        }
        
        self.model_name = model_name
        self.writer = writer
        if self.writer == "auto":
            if model_name is None:
                raise ValueError('model_name is required with writer="auto"')
            self.writer = create_writer(model_name)
            
    def get_last_lr(self) -> float:
        """Get the last used learning rate.
        
        Looks in the scheduler if one is defined, and if not, looks in the
        optimizer. Assumes that a single learning rate is defined for all
        parameters.
        
        Returns:
          The last used learning rate of the trainer.
        """
        if self.scheduler_class is not None:
            return self.scheduler.get_last_lr()[0]
        else:
            return self.optimizer.param_groups[0]['lr']
    
    def train_step(self, X, y):
        self.optimizer.zero_grad()
        outputs = self.model(X.to(DEVICE))
        loss = self.criterion(outputs, y.to(DEVICE))
        loss.backward()
        self.optimizer.step()
        if self.scheduler_class is not None:
            self.scheduler.step()
        return outputs, loss
            
    def val_step(self, X, y):
        with torch.inference_mode():
            val_outputs = self.model(X.to(DEVICE))
            val_loss = self.criterion(val_outputs, y.to(DEVICE))
        return val_outputs, val_loss
    
    def train_eval_loop(self, batches, val_batches, val_period: int = 500,
                        save_period: int | None = None):
        """Train model for a given number of batches, performing validation
        periodically.
        
        Train the model on a number of training batches given by batches. Every
        val_period training batches, pause training and perform validation on
        val_batches batches from the validation set. Each time validation is
        performed, the model's loss, accuracy, and F0.5 scores are saved to the
        trainer, and optionally written to TensorBoard. Optionally, periodically
        save a copy of the model.
        
        Args:
          batches: Number of training batches to use.
          val_batches: Number of validation batches to use each time validation
            is performed.
          val_period: Number of batches to train for in between each occurrence
            of validation (default 500).
          save_period: Number of batches to train for before saving another copy
            of the model (default None, meaning that the model is not saved).
        """
        # Note, this scheduler should not be used if one plans to call
        # train_eval_loop multiple times.
        
        # It doesn't make sense to have more validation steps than batches in
        # the validation set
        val_batches = min(val_batches, len(self.val_loader))
        # estimate total batches
        total_batches = batches + ((batches // val_period) * val_batches)
        pbar = tqdm(total=total_batches, desc="Training")
        
        # Initialize iterator for validation set -- used to continue validation
        # loop from where it left off
        val_iterator = iter(cycle(self.val_loader))
        
        self.model.train()
        total_train_loss = 0
        for i, (X, y) in enumerate(cycle(self.train_loader)):
            pbar.update()
            if i >= batches:
                break
            outputs, loss = self.train_step(X, y)
            total_train_loss += loss.item()
            if (i + 1) % val_period == 0:
                # record number of batches and training metrics
                self.histories['batches'].append(i+1)
                self.histories['train_loss'].append(total_train_loss / val_period)
                total_train_loss = 0

                # record learning rate
                self.histories['lr'].append(self.get_last_lr())

                # predict on validation data and record metrics
                self.model.eval()
                total_val_loss = 0
                for j, (val_X, val_y) in enumerate(val_iterator):
                    pbar.update()
                    if j >= val_batches:
                        break
                    val_outputs, val_loss = self.val_step(val_X, val_y)
                    total_val_loss += val_loss.item()
                self.model.train()
                
                self.histories['val_loss'].append(total_val_loss / j)

                # If logging to TensorBoard, add metrics to writer
                if self.writer is not None:
                    self.writer.add_scalars(
                        main_tag="Loss",
                        tag_scalar_dict={
                            "train_loss": self.histories['train_loss'][-1], 
                            "val_loss": self.histories['val_loss'][-1], 
                        }, 
                        global_step=i+1)
                    self.writer.add_scalar(
                        "Learning rate",
                        self.histories['lr'][-1], 
                        global_step=i+1)
                    # write to disk
                    self.writer.flush()
                    
                # If loss is NaN, the model died and we might as well stop training.
                if np.isnan(self.histories['val_loss'][-1]) or np.isnan(self.histories['train_loss'][-1]):
                    print (f"Model died at training batch {i+1}, stopping training.")
                    break
                    
            # Optionally save a copy of the model
            if save_period is not None and (i + 1) % save_period == 0:
                self.save_checkpoint(f"{i+1}_batches")
                
    def train_loop_simple(self, batches):
        """Train model for a given number of batches.
        
        I made this to diagnose the memory issues. It's an extremely simple
        version of the training loop, without any extra functionality.
        
        Args:
          batches: Number of training batches to use.
        """
        
        self.model.train()
        for i, (X, y) in enumerate(cycle(self.train_loader)):
            if i >= batches:
                break
            self.optimizer.zero_grad()
            outputs = self.model(X.to(DEVICE))
            loss = self.criterion(outputs, y.to(DEVICE))
            loss.backward()
            self.optimizer.step()
            if self.scheduler_class is not None:
                self.scheduler.step()
    
    def time_train_step(self, n=500):
        """Time training on a given number of training batches. Note that this
        does train the model.
        """
        tic = time.perf_counter()
        self.model.train()
        for i in range(n):
            X, y = next(iter(self.train_loader))
            self.train_step(X, y)
        toc = time.perf_counter()
        delta = toc - tic
        print(f"Trained on {n} batches in {delta:.2f}s.")
        return n
        
    def time_val_step(self, n=500):
        """Time prediction on a given number of validation batches."""
        tic = time.perf_counter()
        self.model.eval()
        for i in range(n):
            X, y = next(iter(self.val_loader))
            self.val_step(X, y)
        toc = time.perf_counter()
        delta = toc - tic
        print(f"Predicted on {n} validation batches in {delta:.2f}s.")
        return n

    def plot_metrics(self, ax=None):
        """Plot train and validation loss over time.
        
        Args:
          ax: Axes to plot on (default plt.gca()).
        """
        if ax is None:
            ax = plt.gca()
        plt.plot(self.histories['batches'], self.histories['train_loss'], label="training")
        plt.plot(self.histories['batches'], self.histories['val_loss'], label="validation")
        plt.xlabel('Batch')
        plt.ylabel('loss')
        plt.title("Loss")
        plt.legend()
        
        plt.tight_layout()
        
    def save_checkpoint(self, extra):
        """Save a copy of the model.
        
        The copy will be saved to MODEL_SAVE_DIR/{model_name}_{extra}.pt.
        
        Args:
          extra: String to append to model name to generate filename for saving.
        """
        model_save_path = MODEL_SAVE_DIR / (self.model_name + "_" + extra + ".pt")
        torch.save(self.model.state_dict(), model_save_path)
        print(f"Saved a checkpoint at {model_save_path}.")


### From models.py ###
from torch import nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

class Spectrogram_EfficientNet(nn.Module):
    """An EfficientNetB0 vision model for the spectrogram data."""
    def __init__(self):
        super().__init__()
        self.efficientnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        # freeze pretrained layers besides classifier
        for param in self.efficientnet.features.parameters():
            param.requires_grad = False
        # replace classifier with one of appropriate shape
        self.efficientnet.classifier = nn.Linear(1280, 6)
        # make model output log-probabilities
        self.activation = nn.LogSoftmax(dim=1)
        
    def forward(self, x):
        # Convert grayscale images, [batch, H, W], to RGB images, [batch, 3, H, W]
        return self.activation(self.efficientnet(x.unsqueeze(1).repeat(1, 3, 1, 1)))


### From data_handling.py ###
from pathlib import Path
import os
import numpy as np
import pandas as pd
import joblib
import torch
from tqdm.auto import tqdm


#
# ~~~ Simple function that reads the convents of a .txt file
def read_txt(filepath): # ~~~ Tom added this
    with open(filepath, 'r', encoding='utf-8') as infile:
        return infile.read()

    
class_names = ['Seizure', 'LPD', 'GPD', 'LRDA','GRDA', 'Other']
label2name = dict(enumerate(class_names))
name2label = {v:k for k, v in label2name.items()}

def create_spec_npy_dirs():
    """Create folders to save the spectrogram .npys."""
    os.makedirs(SPEC_DIR/'train_spectrograms', exist_ok=True)
    os.makedirs(SPEC_DIR/'test_spectrograms', exist_ok=True)
    print(f"Created directories {SPEC_DIR/'train_spectrograms'}, {SPEC_DIR/'test_spectrograms'}")

def metadata_df(split="train"):
    """Return a DataFrame with metadata for the train or test set."""
    if split not in ["train", "test"]:
        raise ValueError('Expected split="train" or split="test"')
    metadata = pd.read_csv(f'{BASE_PATH}/{split}.csv')
    metadata['eeg_path'] = f'{BASE_PATH}/{split}_eegs/'+metadata['eeg_id'].astype(str)+'.parquet'
    metadata['spec_path'] = f'{BASE_PATH}/{split}_spectrograms/'+metadata['spectrogram_id'].astype(str)+'.parquet'
    metadata['spec_npy_path'] = f'{SPEC_DIR}/{split}_spectrograms/'+metadata['spectrogram_id'].astype(str)+'.npy'
    if split == "train":
        metadata['class_label'] = metadata.expert_consensus.map(name2label)
    return metadata

def process_spec(spec_id, split="train"):
    """Convert a single spectrogram parquet to .npy, and save the result."""
    if split not in ["train", "test"]:
        raise ValueError('Expected split="train" or split="test"')
    spec_path = f"{BASE_PATH}/{split}_spectrograms/{spec_id}.parquet"
    spec = pd.read_parquet(spec_path)
    spec = spec.fillna(0).values[:, 1:].T # fill NaN values with 0, transpose for (Time, Freq) -> (Freq, Time)
    spec = spec.astype("float32")
    np.save(f"{SPEC_DIR}/{split}_spectrograms/{spec_id}.npy", spec)

def process_all_specs():
    """Convert and save all spectrograms. This could be slow."""
    # Get unique spec_ids of train and valid data
    metadata_train = metadata_df("train")
    spec_ids = metadata_train["spectrogram_id"].unique()

    # Parallelize the processing using joblib for training data
    _ = joblib.Parallel(n_jobs=-1, backend="loky")(
        joblib.delayed(process_spec)(spec_id, "train")
        for spec_id in tqdm(spec_ids, total=len(spec_ids))
    )

    # Get unique spec_ids of test data
    metadata_test = metadata_df("test")
    test_spec_ids = metadata_test["spectrogram_id"].unique()

    # Parallelize the processing using joblib for test data
    _ = joblib.Parallel(n_jobs=-1, backend="loky")(
        joblib.delayed(process_spec)(spec_id, "test")
        for spec_id in tqdm(test_spec_ids, total=len(test_spec_ids))
    )
    print(f"Saved spectrograms as .npy files in {SPEC_DIR}")
    
class SpectrogramDataset(torch.utils.data.Dataset):
    """Dataset with only spectrogram data.
    
    Args:
      metadata_df: DataFrame containing spectrogram offsets, paths to .npy
        files, and vote labels.
      n_items (optional): If not None, generate a dataset with this many items
        by selecting a random subset of metadata_df.
      item_transforms (optional): Transforms to be applied to each item before
        returning it.
      preloaded (default False): whether to load every spectrogram into memory at
        initialization. I estimate that this uses 5-6 GB of RAM and is about
        15% faster during training and 10% faster during inference.
    """
    def __init__(self, metadata_df, n_items=None, item_transforms=None,
                 preloaded=False):
        if n_items is not None:
            self.metadata_df = metadata_df.sample(n=n_items).copy()
        else:
            self.metadata_df = metadata_df
        self.item_transforms = item_transforms
        if preloaded:
            self.spec_dict = {npy_path: torch.from_numpy(np.load(npy_path))
                              for npy_path in self.metadata_df.spec_npy_path.unique()}
        else:
            self.spec_dict = None
        
    def __len__(self):
        return len(self.metadata_df)
    
    def __getitem__(self, i):
        npy_path = self.metadata_df["spec_npy_path"].iloc[i]
        offset = int(self.metadata_df["spectrogram_label_offset_seconds"].iloc[i])
        if self.spec_dict is not None:
            tens = self.spec_dict[npy_path][:, offset//2:offset//2+300]
        else:
            tens = torch.from_numpy(np.load(npy_path))[:, offset//2:offset//2+300]
        if self.item_transforms is not None:
            tens = self.item_transforms(tens)
        expert_votes = self.metadata_df[[
            "seizure_vote", "lpd_vote", "gpd_vote",
            "lrda_vote", "grda_vote", "other_vote"
        ]].iloc[i]
        # target should be float so that nn.KLDivLoss works
        target = torch.tensor(np.asarray(expert_votes)).float()
        return tens, target
    
class SpectrogramTestDataset(SpectrogramDataset):
    def __getitem__(self, i):
        npy_path = self.metadata_df["spec_npy_path"].iloc[i]
        tens = torch.from_numpy(np.load(npy_path))
        target = None
        return tens, target


### From visualization.py ###

def visualize_spectrogram(item, *, figsize=(15, 12), label_spacing=6, colorbars=True):
    """Visualize a spectrogram.
    
    Args:
      item: Spectrogram to visualize, as a np.array or torch.Tensor of shape
        [400, *].
      figsize (default (10, 8)): Size of matplotlib figure.
      label_spacing (default 6): Spacing between y-axis labels.
      colorbars (default True): Whether to include colorbars.
    """
    # Thanks to https://www.kaggle.com/code/alejopaullier/hms-efficientnetb0-pytorch-train#%7C-Utils-%E2%86%91
    item = torch.clamp(item, 1e-4, 1e7)
    item = torch.log(item)
    fig, axs = plt.subplots(2, 2, figsize=figsize)
    axsflat = axs.flatten()
    regions = ["LL", "RL", "LP", "RP"]
    freqs = [0.59, 0.78, 0.98, 1.17, 1.37, 1.56, 1.76, 1.95, 2.15, 2.34, 2.54,
             2.73, 2.93, 3.13, 3.32, 3.52, 3.71, 3.91, 4.1, 4.3, 4.49, 4.69,
             4.88, 5.08, 5.27, 5.47, 5.66, 5.86, 6.05, 6.25, 6.45, 6.64, 6.84,
             7.03, 7.23, 7.42, 7.62, 7.81, 8.01, 8.2, 8.4, 8.59, 8.79, 8.98,
             9.18, 9.38, 9.57, 9.77, 9.96, 10.16, 10.35, 10.55, 10.74, 10.94,
             11.13, 11.33, 11.52, 11.72, 11.91, 12.11, 12.3, 12.5, 12.7, 12.89,
             13.09, 13.28, 13.48, 13.67, 13.87, 14.06, 14.26, 14.45, 14.65, 14.84,
             15.04, 15.23, 15.43, 15.63, 15.82, 16.02, 16.21, 16.41, 16.6, 16.8,
             16.99, 17.19, 17.38, 17.58, 17.77, 17.97, 18.16, 18.36, 18.55, 18.75,
             18.95, 19.14, 19.34, 19.53, 19.73,]
    for i in range(4):
        img = axsflat[i].imshow(item[i*100:(i+1)*100], aspect="auto", origin="lower")
        axsflat[i].set_title(regions[i])
        axsflat[i].set_yticks(np.arange(0, len(freqs), label_spacing))
        axsflat[i].set_yticklabels(freqs[::label_spacing])
        if colorbars:
            cbar = fig.colorbar(img, ax=axsflat[i])
            cbar.set_label('Log(Value)')
        axsflat[i].set_ylabel("Frequency (Hz)")
        axsflat[i].set_xlabel("Time")
    plt.tight_layout()


### From logger.py ###
from pathlib import Path
from datetime import datetime

import numpy as np
import sklearn

import torch
from torch.utils.tensorboard import SummaryWriter



def create_writer(model_name: str) -> SummaryWriter:
    """Create a SummaryWriter instance saving to a specific log_dir.
    
    This allows us to save metric histories, predictions, etc., to TensorBoard.
    log_dir is formatted as logs/YYYY-MM-DD/model_name.
    
    Args:
      model_name: Name of model.
    
    Returns:
      A SummaryWriter object saving to log_dir.
    """
    
    timestamp = datetime.now().strftime("%Y-%m-%d")
    log_dir = LOG_DIR / timestamp / model_name
    print(f"Created SummaryWriter saving to {log_dir}.")
    return SummaryWriter(log_dir=log_dir)
    



