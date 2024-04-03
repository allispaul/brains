
# ~~~ Tom Winckelman wrote this; maintained at https://github.com/ThomasLastName/quality_of_life

import sys
import math
import torch
from tqdm import tqdm

#
# ~~~ Set the random seed for pytorch, numpy, and the base random module all at once
def torch_seed(semilla):    
    torch.manual_seed(semilla)
    if "numpy" in sys.modules.keys():
        sys.modules["numpy"].random.seed(semilla)
    if "random" in sys.modules.keys():
        sys.modules["random"].seed(semilla)

#
# ~~~ Extract the raw tensors from a pytorch Dataset
def convert_Dataset_to_Tensors( object_of_class_Dataset ):
    assert isinstance( object_of_class_Dataset, torch.utils.data.Dataset )
    n_data=len(object_of_class_Dataset)
    return next(iter(torch.utils.data.DataLoader( object_of_class_Dataset, batch_size=n_data )))  # return the actual tuple (X,y)

#
# ~~~ Convert Tensors into a pytorch Dataset; from https://fmorenovr.medium.com/how-to-load-a-custom-dataset-in-pytorch-create-a-customdataloader-in-pytorch-8d3d63510c21
class convert_Tensors_to_Dataset(torch.utils.data.Dataset):
    #
    # ~~~ Define attributes
    def __init__( self, X_tensor, y_tensor, X_transforms_list=None, y_transforms_list=None, **kwargs ):
        super().__init__( **kwargs )
        assert isinstance(X_tensor,torch.Tensor)
        assert isinstance(y_tensor,torch.Tensor)
        assert X_tensor.shape[0]==y_tensor.shape[0]
        self.X = X_tensor
        self.y = y_tensor
        self.X_transforms = X_transforms_list
        self.y_transforms = y_transforms_list
    #
    # ~~~ Method which pytorch requres custom Dataset subclasses to have to enable indexing; see https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
    def __getitem__(self, index):
        x = self.X[index]
        if self.X_transforms is not None:
            for transform in self.X_transforms: 
                x = transform(x)
        y = self.y[index]
        if self.y_transforms is not None:
            for transform in self.y_transforms: 
                y = transform(y)
        return x, y
    #
    # ~~~ Method which pytorch requres custom Dataset subclasses to have; see https://pytorch.org/tutorials/beginner/basics/data_tutorial.html#creating-a-custom-dataset-for-your-files
    def __len__(self):
        return self.y.shape[0]

#
# ~~~ Routine that defines the transform that sends an integer n to the (n+1)-th standard basis vector of R^{n_class} (for n in the range 0 <= n < n_class)
def hot_1_encode_an_integer( n_class, dtype=None ):
    def transform(y,dtype=dtype):
        y = y if isinstance(y,torch.Tensor) else torch.tensor(y)
        y = y.view(-1,1) if y.ndim==1 else y
        dtype = y.dtype if (dtype is None) else dtype
        return torch.zeros( size=(y.numel(),n_class), dtype=dtype ).scatter_( dim=1, index=y.view(-1,1), value=1 ).squeeze()       
    return transform


# todo for DataLoading onto GPU: https://stackoverflow.com/questions/65932328/pytorch-while-loading-batched-data-using-dataloader-how-to-transfer-the-data-t

#
# ~~~ A normal Johnson–Lindenstrauss matrix, which projects to a lower dimension while approximately preserving pairwise distances
def JL_layer( in_features, out_features ):
    linear_embedding = torch.nn.Linear( in_features, out_features, bias=False )
    linear_embedding.weight.requires_grad = False
    torch.nn.init.normal_( linear_embedding.weight, mean=0, std=1/math.sqrt(out_features) )
    return linear_embedding

#
# ~~~ Apply JL "offline" to a dataset
def embed_dataset( dataset, embedding, device=("cuda" if torch.cuda.is_available() else "cpu") ):
    # embedding = JL_layer(D,d).to(device) if embedding is None else embedding
    batches_of_data = torch.utils.data.DataLoader( dataset, batch_size=100, shuffle=False )
    for j, (X,y) in enumerate(tqdm(batches_of_data)):
        X = X.to(device)
        y = y.to(device)
        if j==0:
            embedded_X = embedding(X)
            the_same_y = y
        else:
            embedded_X = torch.row_stack(( embedded_X, embedding(X) ))
            the_same_y = torch.row_stack(( the_same_y, y ))
    return embedded_X, the_same_y


# TODO: faster is to simply generate `embedding=torch.randn(d,D,device="cuda")>0`
#       if you just torch.save(embedding,"boolean.pt") it will be memory efficient, as will torch.load("boolean.pt")
#       to convert to float, you need to three separate lines:
#           embedding = embedding.float()
#           embedding *= 2
#           embedding -= 1
#       in contrast, embedding=2*embedding.float()-1 gives a memory error
#       on the other hand, e_float@x == ( (x*e_bool) - (x*~e_bool) ).sum(axis=1) when e_bool = torch.randn(d,D,device="cuda")>0 and e_float = 2*e_bool-1.
#       e_float@x is perhaps faster, but the latter is more memory efficient, as e_float never needs to be loaded in memory
