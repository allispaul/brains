
import sys
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch
from brains.utils.data_handling import metadata_df
from brains.utils.data_handling import SPEC_DIR
try:
    from tabulate import tabulate   # ~~~ use if available
except:
    pass

metadata = metadata_df("train")

def get_item( id, item="spectrogram", metadata=metadata, center=True, to_tensor=True ):
    #
    # ~~~ Get either the spectrogram, the eeg, or the eeg with timestamps, depending on the `item` arg
    assert item=="spectrogram" or item=="eeg" or item=="raw"
    if item=="spectrogram":
        got = np.load(metadata["spec_npy_path"].iloc[id])
    else:
        got = pd.read_parquet(metadata.iloc[id].eeg_path)
        # ~~~ pd.read_parquet(metadata.iloc[id].spec_path) would analogously give the raw spctrogram data
        if item=="eeg":
            got = got.values
    #
    # ~~~ Crop for only the time frame as in the image shown here https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/468010
    offset = None if item=="raw" else int(metadata[f"{item}_label_offset_seconds"].iloc[id])
    if center and item!="raw":
        start = offset//2
        width = 300 if item=="spectrogram" else 25
        got = got[:, start:(start+width)]
    #
    # ~~~ Gather what the experts voted for this item
    votes = np.asarray(metadata[[
            "seizure_vote", "lpd_vote", "gpd_vote",
            "lrda_vote", "grda_vote", "other_vote"
        ]].iloc[id])
    label = metadata["expert_consensus"].iloc[id]
    #
    # ~~~ Convert to torch.tensors if desired
    if to_tensor and item!="raw":
        got = torch.from_numpy(got)
        votes = torch.from_numpy(votes)
    #
    # ~~~ Return (i) the data, (ii) what the experts voted, (iii) the result of the vote, and (iv) the index that we started with
    return got, votes, label, offset


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
    def __init__( self, metadata_df, n_items=None, item_transforms=None, preloaded=False ):
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
    


def get_spec( *args, **kwargs ):
    return get_item( *args, item="spectrogram", **kwargs )


def get_eeg( *args, **kwargs ):
    return get_item( *args, item="eeg", **kwargs )


def get_raw( *args, **kwargs ):
    return get_item( *args, item="raw", **kwargs )

#
# ~~~ Build the path to the .npy file from the spec_id
def id_to_npy_path( spec_id, path=SPEC_DIR ):
    path /= "train_spectrograms"
    path /= f"{spec_id}.npy"
    return path

load_id = lambda spec_id: np.load( id_to_npy_path(spec_id) )

def plot_spec( id, metadata=metadata ):
    spectrogram,_,_,offset = get_spec( id, metadata=metadata, to_tensor=False )
    img = np.log1p(spectrogram)
    img -= img.min()
    img /= img.max() + 1e-4
    plt.imshow(img)
    plt.xticks( np.arange(0,300,50), np.arange(offset,offset+600,100) )
    plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


def plot_eeg( id, metadata=metadata ):
    eeg,_,_,_ = get_eeg( id, metadata=metadata, to_tensor=False )
    plt.figure(figsize=(12,4))
    plt.plot(eeg[:500])
    plt.tight_layout()
    plt.show()


#
# ~~~ View a table of the votes by time for a particular spectrogram id
def view_votes( spec_id, metadata=metadata, print_it=True ):
    data_on_this_spectrogram = metadata[metadata.spectrogram_id==spec_id]
    expert_votes = data_on_this_spectrogram[[ "spectrogram_label_offset_seconds", "expert_consensus", "seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote" ]]
    col_names = list(expert_votes.columns)
    col_names[0] = "offset (sec)"
    col_names[1] = "consensus"
    if print_it:
        tabulate_is_available = "tabulate" in sys.modules.keys()
        if tabulate_is_available:
            expert_votes = sys.modules["tabulate"].tabulate( expert_votes, headers=col_names, tablefmt="github", showindex=False )
        else:
            warnings.warn("For neater printing, install the `tabulate` module.")
            expert_votes = expert_votes.to_string(index=False)
        print(expert_votes)
    else:
        return expert_votes


def train_val_split( metadata, val_frac=0.2, seed=4, verbose=True ):
    #
    # ~~~ Count the "true" sizes of our datasets (each case has multiple spectrograms, and we want "a datapoint" to be a complete case, not just 1 spectrogram)
    cases = metadata.spectrogram_id.unique()    # ~~~ a "single case" is represented my multiple "rows" of data
    n_cases = len(cases)            # ~~~ number of different cases represented by our training set
    n_val = round(val_frac*n_cases) # ~~~ number of different cases that we want in the validation set
    if verbose:
        print("")
        print(f"{n_cases} unique spectrograms, using {n_val} for validation set.")
    #
    # ~~~ Split the data not by "rows" but, rather, in a manner that respects the "grouped" nature of our data
    rng = np.random.default_rng(seed=seed)
    val_idx = rng.choice( cases, size=n_val, replace=False )
    metadata_train = metadata[~metadata.spectrogram_id.isin(val_idx)]
    metadata_valid = metadata[metadata.spectrogram_id.isin(val_idx)]
    if verbose:
        print("")
        print(f"{len(metadata_train)} training items, {len(metadata_valid)} validation items.")
        print("")
    return metadata_train, metadata_valid, val_idx
