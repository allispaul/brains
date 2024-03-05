


import math
import random
import numpy as np
import pandas as pd
import torch
from brains.utils.data_handling import metadata_df
from brains.utils import SpectrogramDataset
from sklearn.naive_bayes import GaussianNB
torch.set_default_dtype(torch.double)

#
# ~~~ Copied from https://github.com/ThomasLastName/quality_of_life/blob/main/my_torch_utils.py
def convert_Dataset_to_Tensors( object_of_class_Dataset ):
    assert isinstance( object_of_class_Dataset, torch.utils.data.Dataset )
    n_data=len(object_of_class_Dataset)
    return next(iter(torch.utils.data.DataLoader( object_of_class_Dataset, batch_size=n_data )))  # return the actual tuple (X,y)

#
# ~~~ Make a subset of the data consting of only n items
n = 5000
hold_out = 0.2
metadata = metadata_df("train")
subset = SpectrogramDataset(metadata,n_items=n)
X, y = convert_Dataset_to_Tensors(subset) # ~~~ indirect and inefficient

#
# ~~~ Process the data
expert_consensus = y.argmax(dim=1).to(X.dtype)
spectrograms = X.reshape((X.shape[0],-1))
indices_for_training = torch.rand(n)>hold_out
X_train, y_train = spectrograms[indices_for_training], expert_consensus[indices_for_training]
X_test,  y_test  = spectrograms[~indices_for_training], expert_consensus[~indices_for_training]
print(f"Using: {len(X_train)} spectrograms for training and {len(X_test)} spectrograms for testing.")


#
X_train = X_train.numpy()
y_train = y_train.numpy()
X_test = X_test.numpy()
y_test = y_test.numpy()

gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
acc = np.sum(y_pred==y_test) / len(y_test)
print(f"Classifier accuracy: {acc}")

#
# ~~~ Try learning the function p_k(x) = p(x|C_k) directly

indices_for_training = torch.rand(n)>hold_out
X_train, y_train = X[indices_for_training],  y[indices_for_training]
X_test,  y_test  = X[~indices_for_training], y[~indices_for_training]

D = X_train.shape[1]
d = 20000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
A_JL = torch.randn( size=(d,D), dtype=X.dtype, device=DEVICE )/math.sqrt(d)
X_train = X_train.reshape(( X_train.shape[0], -1 )).to(DEVICE)
X_test  = X_test.reshape(( X_test.shape[0], -1 )).to(DEVICE)
X_train_JL = (A_JL @ X_train.T).cpu()
X_test_JL  = (A_JL @ X_test.T).cpu()

#
X_train = X_train.numpy().reshape(( X_train.shape[0], -1 ))
y_train = y_train.numpy()
X_test = X_test.numpy().reshape(( X_test.shape[0], -1 ))
y_test = y_test.numpy()





import tensorflow as tf

# Define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(400, 300, 1)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',  # For multi-class classification
              metrics=['accuracy'])

# Expand the dimensions of X_train to make it compatible with Conv2D
X_train_expanded = X_train.reshape(-1, 400, 300, 1)

# Train the model
model.fit(X_train_expanded, y_train, epochs=10, batch_size=32, verbose=1)