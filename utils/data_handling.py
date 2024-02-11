"""Code to produce metadata files and to convert spectrogram .parquet files to
.npy. A lot of this is from the Starter Notebook on Kaggle.

You should make sure that BASE_PATH and SPEC_DIR are set correctly before
running these functions.
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd
import joblib
from tqdm.auto import tqdm

# BASE_PATH = Path("/kaggle/input/hms-harmful-brain-activity-classification")
BASE_PATH = Path("data/")
# SPEC_DIR = Path("/tmp/dataset/hms-hbac")
SPEC_DIR = Path("data/spectrograms_npy")

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