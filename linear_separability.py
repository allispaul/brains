
import math
import pandas as pd
import torch
from brains.utils.data_handling import metadata_df
from brains.utils import SpectrogramDataset


### ~~~
## ~~~ A simple implementation of the perceptron algorithm (Foucart ch 4), which converges if run on linearly separable data
### ~~~


def preceptron_update_without_bias(X,y,w,verbose=True):
    pred = torch.sign(X @ w)
    misclassified_mask = (pred != torch.sign(y))
    if not misclassified_mask.any():  # if all True
        raise StopIteration
    else:
        misclassified_indices = torch.nonzero(misclassified_mask).squeeze()
        i = misclassified_indices[0].item()  # Find the index of the first misclassified sample
        x_i = X[i, :]  # the misclassified data point
        new_w = w + y[i] / torch.inner(x_i, x_i) * x_i  # the update rule
    if verbose:
        print(f"Accuracy: {len(misclassified_indices)/len(y)}")
    return new_w

#
# ~~~ Copied from https://github.com/ThomasLastName/quality_of_life/blob/main/my_numpy_utils.py
def augment(X):
    return torch.hstack((
            X,
            torch.ones( size=(X.shape[0],1), dtype=X.dtype )  # assumes that X has dtype and shape attributes
        ))

#
# ~~~ Basic loop
def naive_perceptron( X_train, y_train, verbose=True ):
    y = y_train.to(X.dtype)
    assert set(y.tolist())=={-1,1}
    _,d = X.shape
    w = torch.randn( size=(d,), dtype=X.dtype, device=X.device )/math.sqrt(d)
    i=0
    while True:
        i+=1
        if verbose:
            # print(i)
            pass
        try:
            w = preceptron_update_without_bias( X_train, y_train, w, verbose=verbose )
        except StopIteration:
            break
    return w



### ~~~
## ~~~ Fetch some data
### ~~~

#
# ~~~ Copied from https://github.com/ThomasLastName/quality_of_life/blob/main/my_torch_utils.py
def convert_Dataset_to_Tensors( object_of_class_Dataset ):
    assert isinstance( object_of_class_Dataset, torch.utils.data.Dataset )
    n_data=len(object_of_class_Dataset)
    return next(iter(torch.utils.data.DataLoader( object_of_class_Dataset, batch_size=n_data )))  # return the actual tuple (X,y)

#
# ~~~ Make a subset of the data consting of only n items
n=10000
metadata = metadata_df("train")
subset = SpectrogramDataset(metadata,n_items=n)
X_train,y_train = convert_Dataset_to_Tensors(subset) # ~~~ indirect and inefficient

#
# ~~~ Convert to integer labels based on majority vote
y_train = y_train.argmax(dim=1)

#
# ~~~ Reduce to a further subset of the data by discarding all except the most common two classes
most_common_class = y_train.mode().values.item()
second_most_common = y_train[~(y_train==most_common_class)].mode().values.item()
indecies_of_most_common_two_classes = torch.max(y_train==most_common_class, y_train==second_most_common)
y = y_train[indecies_of_most_common_two_classes]
X = X_train[indecies_of_most_common_two_classes]

#
# ~~~ Normalize the labels to be {-1,1}
lo = min(most_common_class,second_most_common)
hi = max(most_common_class,second_most_common)
y = (y-lo)/(hi-lo)*2-1

#
# ~~~ Flatten X and augment with a row of all 1's
X = augment( X.reshape((X.shape[0],-1)) )

#
# ~~~ Move to GPU
device = "cuda" if torch.cuda.is_available() else "cpu"
X = X.to(device)
y = y.to(device)



### ~~~
## ~~~ Run the perceptron algorithm
### ~~~

naive_perceptron(X,y)

#