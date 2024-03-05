
import warnings
import math
import torch
import torchvision
from torch import nn
from tqdm import tqdm
from quality_of_life.my_torch_utils import fit
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

metadata = metadata_df("train")
metadata_train, _, val_idx = train_val_split( metadata, val_frac=0.1, seed=4, verbose=True )
unsqueeze = lambda x: x.unsqueeze(dim=0)/2000
brains_train = SpectrogramDataset( metadata_train, item_transforms=unsqueeze, normalize=True )
# brains_valid = SpectrogramDataset( metadata_valid, item_transforms=unsqueeze )
b = 50
training_batches = torch.utils.data.DataLoader( brains_train, batch_size=b, shuffle=True )



### ~~~
## ~~~ Compute the empirical mean and standard deviation of our data for standardization
### ~~~

# e_mean = torch.zeros_like(brains_train[0][0]).to(device)
# e_2nd_moment = torch.zeros_like(brains_train[0][0]).to(device)
# for X,_ in tqdm(training_batches):
#     X = X.to(device)
#     e_mean += X.mean(dim=0)
#     e_2nd_moment += (X**2).sum(dim=0)

# e_std_dev = torch.sqrt(e_2nd_moment - e_mean**2)



### ~~~
## ~~~ Build the network
### ~~~

#
# ~~~ Set up a Johnson-Lindenstrauss embedding to reduce the dimension of the data
D = math.prod(list(brains_train[0][0].shape))   # ~~~ should be 400*300==120000
d = 10000                                       # ~~~ desired lower dimension
# JL_layer = nn.Linear(400*300,d)
# JL_layer.weight.requires_grad = False
# JL_layer.bias.requires_grad = False
# torch.nn.init.normal_( JL_layer.weight, mean=0, std=1/math.sqrt(D) )
# JL_layer.bias.zero_()

#
# ~~~ Instantiate
model = nn.Sequential(
            nn.Flatten(),
            # JL_layer,
            nn.Linear(D, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
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

summary(model)

def measure_accuracy( model, data, device=device ):
    X = data[0].to(device)
    y = data[1].to(device)
    p = model(X).argmax(dim=1)
    t = y.argmax(dim=1)
    batch_size = len(y)
    n_correct = torch.sum( t==p ).item()
    return n_correct/batch_size

history = fit(
        model,
        training_batches=torch.utils.data.DataLoader(brains_train,batch_size=b),
        loss_fn=nn.KLDivLoss( reduction="batchmean" ), #,
        optimizer=torch.optim.Adam( model.parameters(), lr=1e-6 ),
        training_metrics = { "accuracy": measure_accuracy },
        epochs = 10
    )
