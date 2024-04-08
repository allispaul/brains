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
import torch
from tqdm.auto import tqdm

from .config import BASE_PATH, SPEC_DIR, EEG_DIR

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
    train_spec_path = SPEC_DIR/'train_spectrograms'
    test_spec_path = SPEC_DIR/'test_spectrograms'
    if os.path.exists(train_spec_path):
        print(f"Directory {train_spec_path} already existed")
    else:
        os.makedirs(train_spec_path)
        print(f"Created directory {train_spec_path}")
    if os.path.exists(test_spec_path):
        print(f"Directory {test_spec_path} already existed")
    else:
        os.makedirs(test_spec_path)
        print(f"Created directory {test_spec_path}")
        
def create_eeg_npy_dirs():
    """Create folders to save the eeg .npys."""
    train_eeg_path = EEG_DIR/'train_eegs'
    test_eeg_path = EEG_DIR/'test_eegs'
    if os.path.exists(train_eeg_path):
        print(f"Directory {train_eeg_path} already existed")
    else:
        os.makedirs(train_eeg_path)
        print(f"Created directory {train_eeg_path}")
    if os.path.exists(test_eeg_path):
        print(f"Directory {test_eeg_path} already existed")
    else:
        os.makedirs(test_eeg_path)
        print(f"Created directory {test_eeg_path}")
    
def metadata_df(split="train"):
    """Return a DataFrame with metadata for the train or test set."""
    if split not in ["train", "test"]:
        raise ValueError('Expected split="train" or split="test"')
    metadata = pd.read_csv(f'{BASE_PATH}/{split}.csv')
    metadata['eeg_path'] = f'{BASE_PATH}/{split}_eegs/'+metadata['eeg_id'].astype(str)+'.parquet'
    metadata['spec_path'] = f'{BASE_PATH}/{split}_spectrograms/'+metadata['spectrogram_id'].astype(str)+'.parquet'
    metadata['spec_npy_path'] = f'{SPEC_DIR}/{split}_spectrograms/'+metadata['spectrogram_id'].astype(str)+'.npy'
    metadata['eeg_npy_path'] = f'{EEG_DIR}/{split}_eegs/'+metadata['eeg_id'].astype(str)+'.npy'
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
    
def process_eeg(eeg_id, split="train"):
    """Convert a single eeg parquet to .npy, and save the result."""
    eeg_path = f"{BASE_PATH}/{split}_eegs/{eeg_id}.parquet"
    eeg = pd.read_parquet(eeg_path)
    eeg = eeg.fillna(0).values[:, :].T # fill NaN values with 0, transpose for (Time, Amplitude) -> (Amplitude, Time)
    eeg = eeg.astype("float32")
    np.save(f"{EEG_DIR}/{split}_eegs/{eeg_id}.npy", eeg)

def process_all_eegs():
    """Convert and save all eegs. This could be slow."""
    # Get unique eeg_ids of train and valid data
    metadata_train = metadata_df("train")
    eeg_ids = metadata_train["eeg_id"].unique()

    # Parallelize the processing using joblib for training data
    _ = joblib.Parallel(n_jobs=-1, backend="loky")(
        joblib.delayed(process_eeg)(eeg_id, "train")
        for eeg_id in tqdm(eeg_ids, total=len(eeg_ids))
    )

    # Get unique eeg_ids of test data
    metadata_test = metadata_df("test")
    test_eeg_ids = metadata_test["eeg_id"].unique()

    # Parallelize the processing using joblib for test data
    _ = joblib.Parallel(n_jobs=-1, backend="loky")(
        joblib.delayed(process_eeg)(eeg_id, "test")
        for eeg_id in tqdm(test_eeg_ids, total=len(test_eeg_ids))
    )
    print(f"Saved eegs as .npy files in {EEG_DIR}")
    
    
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
      dtype (optional): torch.float datatype to use.
      random_state (optional): random seed to use when sampling items, if
        n_items is not None.
    """
    def __init__(
            self,
            metadata_df,
            n_items=None,
            item_transforms=None,
            preloaded=False,
            normalize_targets=True,
            dtype=None,
            random_state=None,
        ):
        if n_items is not None:
            self.metadata_df = metadata_df.sample(n=n_items, random_state=random_state).copy()
        else:
            self.metadata_df = metadata_df
        self.item_transforms = item_transforms
        self.normalize_targets = normalize_targets
        if dtype is None:
            self.dtype = torch.get_default_dtype()
        else:
            self.dtype = dtype
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
        target = torch.tensor(np.asarray(expert_votes)).float()
        if self.normalize_targets:
            target /= target.sum()
        # target should be float so that nn.KLDivLoss works
        return tens.to(self.dtype), target.to(self.dtype)
    
    
class SpectrogramTestDataset(SpectrogramDataset):
    def __getitem__(self, i):
        npy_path = self.metadata_df["spec_npy_path"].iloc[i]
        tens = torch.from_numpy(np.load(npy_path))
        target = torch.tensor([1., 1., 1., 1., 1.])  # dummy values
        return tens, target
    
    
class EegDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            metadata_df,
            n_items=None,
            item_transforms=None,
            preloaded=False,
            force_unique=False,
            normalize_targets=True,
            dtype=None,
            random_state=None,
        ):
        self.metadata_df = metadata_df.copy()
        if force_unique:
            # Select a unique row for each value of eeg_id
            selected_rows = [
                metadata_df[metadata_df['eeg_id'] == eeg_id].sample(1, random_state=random_state)
                for eeg_id in metadata_df['eeg_id'].unique()
            ]
            self.metadata_df = pd.concat(selected_rows, ignore_index=True)
        if n_items is not None:
            self.metadata_df = self.metadata_df.sample(n=n_items, random_state=random_state)
        self.item_transforms = item_transforms
        self.normalize_targets = normalize_targets
        if dtype is None:
            self.dtype = torch.get_default_dtype()
        else:
            self.dtype = dtype
        if preloaded:
            self.eeg_dict = {npy_path: torch.from_numpy(np.load(npy_path))
                              for npy_path in self.metadata_df.eeg_npy_path.unique()}
        else:
            self.eeg_dict = None
        
    def __len__(self):
        return len(self.metadata_df)
    
    def __getitem__(self, i):
        npy_path = self.metadata_df["eeg_npy_path"].iloc[i]
        offset = 200*int(self.metadata_df["eeg_label_offset_seconds"].iloc[i])
        #CHANGE THIS LINE PROPERLY
        if self.eeg_dict is not None:
            tens = self.eeg_dict[npy_path][:, offset:offset+10000]
        else:
            tens = torch.from_numpy(np.load(npy_path))[:, offset:offset+10000]
        if self.item_transforms is not None:
            tens = self.item_transforms(tens)
        expert_votes = self.metadata_df[[
            "seizure_vote", "lpd_vote", "gpd_vote",
            "lrda_vote", "grda_vote", "other_vote"
        ]].iloc[i]
        target = torch.tensor(np.asarray(expert_votes)).float()
        if self.normalize_targets:
            target /= target.sum()
        # target should be float so that nn.KLDivLoss works
        return tens.to(self.dtype), target.to(self.dtype)
    
    
class EegTestDataset(EegDataset):
    def __getitem__(self, i):
        npy_path = self.metadata_df["eeg_npy_path"].iloc[i]
        tens = torch.from_numpy(np.load(npy_path))
        target = torch.tensor([1., 1., 1., 1., 1.])  # dummy values
        return tens, target
