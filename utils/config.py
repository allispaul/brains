from pathlib import Path
import torch

from .kaggle_platform import this_is_running_in_kaggle

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