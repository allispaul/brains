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

from .config import BASE_PATH, SPEC_DIR

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
    already_exists_train = os.path.exists(SPEC_DIR/'train_spectrograms')
    already_exists_test = os.path.exists(SPEC_DIR/'test_spectrograms')
    #
    # ~~~ Paul's code
    os.makedirs(SPEC_DIR/'train_spectrograms', exist_ok=True)
    os.makedirs(SPEC_DIR/'test_spectrograms', exist_ok=True)
    # ~~~ End Paul's code
    #
    # ~~~ Tom added the following print logic
    if already_exists_train:
        print(f"Directory {SPEC_DIR/'train_spectrograms'} already existed")
    else:
        print(f"Created directory {SPEC_DIR/'train_spectrograms'}")
    if already_exists_train:
        print(f"Directory {SPEC_DIR/'test_spectrograms'} already existed")
    else:
        print(f"Created directory {SPEC_DIR/'test_spectrograms'}")

def metadata_df(split="train"):
    """Return a DataFrame with metadata for the train or test set."""
    if split not in ["train", "test"]:
        raise ValueError('Expected split="train" or split="test"')
    metadata = pd.read_csv(f'{BASE_PATH}/{split}.csv')
    metadata['eeg_path'] = f'{BASE_PATH}/{split}_eegs/'+metadata['eeg_id'].astype(str)+'.parquet'
    # metadata['eeg_path'] = [os.path.join( BASE_PATH, f"{split}_eegs", f"{id}.parquet" ) for id in metadata['eeg_id']]
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
    def __init__(
            self,
            metadata_df,
            n_items=None,
            item_transforms=None,
            preloaded=False,
            normalize_targets=False,
            dtype =  torch.get_default_dtype()
        ):
        if n_items is not None:
            self.metadata_df = metadata_df.sample(n=n_items).copy()
        else:
            self.metadata_df = metadata_df
        self.item_transforms = item_transforms
        self.normalize_targets = normalize_targets
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
        # target should be float so that nn.KLDivLoss works
        target = torch.tensor(np.asarray(expert_votes)).float()
        if self.normalize_targets:
            target /= target.sum()
        return tens.to(self.dtype), target.to(self.dtype)
    
class SpectrogramTestDataset(SpectrogramDataset):
    def __getitem__(self, i):
        npy_path = self.metadata_df["spec_npy_path"].iloc[i]
        tens = torch.from_numpy(np.load(npy_path))
        target = None
        return tens, target
