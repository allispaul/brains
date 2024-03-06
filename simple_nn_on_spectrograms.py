
import warnings
import math
import torch
import torchvision
from torch import nn
from tqdm import tqdm
from quality_of_life.my_torch_utils import fit, JL_layer, convert_Dataset_to_Tensors  # ~~~ not reproducible... I might or might not fix this
from brains.utils import SpectrogramDataset
from brains.utils.data_handling import metadata_df
from brains.Toms_utils import train_val_split
from torchinfo import summary

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(3042024)


### ~~~
## ~~~ Gather data
### ~~~

#
# ~~~ Load the data
metadata = metadata_df("train")
metadata_train, metadata_test, val_idx = train_val_split( metadata, val_frac=0.05, seed=4, verbose=True )

#
# ~~~ Define a pre-processing step
D = 120000  # ~~~ the dimension 400*300 of one spectrogram
d = 10000   # ~~~ a desired smaller dimension
embed = JL_layer(D,d).to(device)
def flatten_standardize_and_embed(x):
    x = x.flatten().to(device)                      # ~~~ flatten the matrix
    standardized = (x-x.mean(dim=-1))/x.std(dim=-1) # ~~~ standardize it
    return embed(standardized.to(device))           # ~~~ embed into the lower dimension d

#
# ~~~ Set up the dataset to include the pre-processing step
brains_train = SpectrogramDataset( metadata_train, item_transforms=flatten_standardize_and_embed, normalize_targets=True, preloaded=True )
test_data = convert_Dataset_to_Tensors(SpectrogramDataset( metadata_test, item_transforms=flatten_standardize_and_embed, normalize_targets=True, preloaded=True ))
X_test, y_test = test_data
X_test = X_test.to(device)
y_test = y_test.to(device)

### ~~~
## ~~~ Build the network
### ~~~

#
# ~~~ Instantiate
model = nn.Sequential(
            nn.Linear(d, 5000),
            nn.ReLU(),
            nn.Linear(5000, 500),
            nn.ReLU(),
            nn.Linear(500, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 6),
            nn.LogSoftmax(dim=1)
        ).to(device)

def measure_accuracy( model, data, device=device ):
    X = data[0].to(device)
    y = data[1].to(device)
    p = model(X).argmax(dim=1)
    t = y.argmax(dim=1)
    batch_size = len(y)
    n_correct = torch.sum( t==p ).item()
    return n_correct/batch_size


summary(model)

history = fit(
        model,
        training_batches = torch.utils.data.DataLoader( brains_train, batch_size=100, shuffle=True ),
        test_data = (X_test,y_test),
        loss_fn = nn.KLDivLoss( reduction="batchmean" ),
        optimizer = torch.optim.Adam( model.parameters(), lr=1e-2 ),
        training_metrics = { "train batch acc": measure_accuracy },
        test_metrics = { "test set acc": measure_accuracy },
        epochs = 1
    )
