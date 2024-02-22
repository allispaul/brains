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

import warnings
# Ignore PyTorch's KLDivLoss warning
warnings.simplefilter("ignore", category=UserWarning, lineno=2949)

from tqdm.auto import tqdm

from .logger import create_writer
from .config import DEVICE, MODEL_SAVE_DIR


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