"""Tools for logging model performance to Tensorboard."""
from pathlib import Path
from datetime import datetime

import numpy as np
import sklearn

import torch
from torch.utils.tensorboard import SummaryWriter


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
    log_dir = Path("logs") / timestamp / model_name
    print(f"Created SummaryWriter saving to {log_dir}.")
    return SummaryWriter(log_dir=log_dir)
    
