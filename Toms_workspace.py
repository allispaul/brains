
### ~~~
## ~~~ Import block
### ~~~

#
# ~~~ Base packages
from time import time
from pathlib import Path
import random
import warnings
import os
import warnings
# Do not ignore pandas deprecation warning
# warnings.filterwarnings("ignore", "\nPyarrow", DeprecationWarning)

#
# ~~~ Standard packages
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas as pd
import torch
from tqdm import trange

#
# ~~~ Custom packages
from brains.utils.data_handling import metadata_df, process_all_specs, create_spec_npy_dirs, BASE_PATH, SPEC_DIR    # ~~~ this works
from brains.utils import SpectrogramDataset, SpectrogramTestDataset
from brains.Toms_utils import *



### ~~~
## ~~~ Convert the spectrograms to .npy (only needs to be run once, then it's done forever)
### ~~~

if False:
    create_spec_npy_dirs()
    process_all_specs()



### ~~~
## ~~~ Load the data
### ~~~

metadata = metadata_df("train")
metadata_test = metadata_df("test")

#
# ~~~ Validate that it was loaded successfully by reproducing the first image in Paul's notebook
assert 277==(metadata.expert_consensus == 'LPD').argmax()   # ~~~ 277 is the first index of any eeg with expert_consensus=='LPD'
plot_eeg(277)

#
# ~~~ Verify that metadata is as discussed in https://www.kaggle.com/competitions/hms-harmful-brain-activity-classification/discussion/468010
assert len(metadata)==106800                            # ~~~ "We are given train.csv with 106,800 rows...
assert len(metadata["eeg_id"].unique())==17089          # ... but there are only 17089 unique eeg_ids, ...
assert len(metadata["spectrogram_id"].unique())==11138  # ... 11138 unique spectrogram_ids, ...
assert len(metadata["patient_id"].unique())==1950       # ... and 1950 unique patients."



### ~~~
## ~~~ Question: Does the expert consensus depend only on the spectrogram id? Or can it also change depending on the time offset? (Answer: it may depend on the time offset)
### ~~~

try:
    for spec_id in metadata.spectrogram_id.unique():
        assert 1 == metadata[metadata.spectrogram_id==spec_id].expert_consensus.nunique()
    print("The expert_consensus depends only on spectrogram_id")
except:
    print("The expert_consensus for the same spectrogram_id can have different values for different time offsets")

#
# ~~~ View the tableu of votes of the spec_id for which the `except` clause triggered
view_votes( spec_id )



### ~~~
## ~~~ Question: is every X item in our dataset just a submatrix resulting from taking certain columns of a spectrogram?
### ~~~

all_data = SpectrogramDataset(metadata)
row = 1
x_paul,_ = all_data[row]
x = np.load( metadata.spec_npy_path.iloc[row])
np.equal( x[:,:300], x_paul.numpy() ).all()


# look at where independence is used in McDiarmid
# look at the compressive sensing paper https://arxiv.org/abs/1310.5791
# What if a(i) and b(i) are \psi_4 in the expresion \|\cala(uv^*)\^2 = \sum_{i \leq m} |v^*a(i)|^2 |u^*b(i)|^2?
# Can we use the "probable and favorable" approach to showing that |\langle b(i), v \rangle| is probably about 1?
# near-optimality of linear recovery in gaussian observation scheme under \ell^2 loss https://arxiv.org/abs/1602.01355

9

# hypoethsis
# to get the i-th data point in all_data
# look at the i-th row of the metadata dataframe
# get the spec_id in that row (the i-th spec_id) metadata.spectrogram_id.iloc[i]
# get the path to that .npy: id_to_npy_path(metadata.spectrogram_id.iloc[i]) (or in hindsight just metadata.spec_npy_path.iloc[i])
# get the i-th offset offset = metadata.spectrogram_label_offset_seconds.iloc[i]
# divide it by 2 to get first_col=offset/2
# take columns first_col:first_col+300 from the .npy so like np.load(path_i)[ : , first_col:first_col+300 ]

def get_ith_data_pair(i,metadata=metadata):
    spectrogram = np.load(metadata.spec_npy_path.iloc[i])   # ~~~ each column corresponds to one point in time
    time_offset = metadata.spectrogram_label_offset_seconds.iloc[i]
    first_col = int(time_offset//2)
    x = spectrogram[ :, first_col:(first_col+300)]       # ~~~ 300 of the spectrogram's columns starting at first_col
    y = metadata.iloc[i][[ "seizure_vote", "lpd_vote", "gpd_vote", "lrda_vote", "grda_vote", "other_vote" ]].values
    return x, np.array(y, dtype=x.dtype )

#
# ~~~ test my hypothesis
for i in trange(len(all_data)):
    x_paul,y_paul = all_data[i]
    x,y = get_ith_data_pair(i)
    assert np.equal( x, x_paul.numpy() ).all()  # ~~~ assert that x==x_paul

#
# ~~~ Also note the difference in speed
tick = time()
for i in range(300):
    x,y = get_ith_data_pair(i)

tock = time()-tick

tick = time()
for i in range(300):
    x_paul,y_paul = all_data[i]

tock_paul = time()-tick

print(f"get_ith_data_pair() is {tock_paul/tock} times faster than the __getitem__() method")

### ~~~
## ~~~ Process the data for use in machine learning
### ~~~

#
# ~~~ Demonstration
id = 4  # ~~~ or any other integer in range(len(metadata))
foo = pd.read_parquet(metadata.iloc[id].spec_path).values[:,1:].T
bar = np.load(metadata["spec_npy_path"].iloc[id])
assert abs(foo - bar).max() < 1e-15

metadata_train, metadata_valid, val_idx = train_val_split( metadata, val_frac=0.1, seed=4, verbose=True )
train = SpectrogramDataset(metadata_train)
valid = SpectrogramDataset(metadata_valid)
test = SpectrogramTestDataset(metadata_test)
#

X_paul, y_paul = train[0]   # ~~~ The first X and y values in the training dataset
X,y,c,t0 = get_spec(0)
assert abs(X_paul-X).max()==0


all_data = SpectrogramDataset(metadata,preloaded=True)

# for key, value, i in enumerate(all_data.spec_dict.items()):
#     npy_path = key
#     offset = int(all_data.metadata["spectrogram_label_offset_seconds"].iloc[i])
#         if self.spec_dict is not None:
#             tens = self.spec_dict[npy_path][:, offset//2:offset//2+300]

