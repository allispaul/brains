
from pyarrow import parquet as pq
import numpy as np
import pandas as pd
import time
from tqdm import tqdm

from brains.utils.data_handling import metadata_df

metadata = metadata_df("train")
spec_paths = metadata.spec_path.unique()
spec_npy_paths = metadata.spec_npy_path.unique()
full_test = False

if not full_test:
    #
    # ~~~ Method 1: Direct conversion to NumPy array
    start_time = time.time()
    table = pq.read_table("C:\\Users\\thoma\\AppData\\Local\\Programs\\Python\\Python310\\Custom\\brains\\data\\train_spectrograms\\353733.parquet")
    numpy_array_direct = np.array(list(table.to_pydict().values()), dtype=float)
    direct_conversion_time = time.time() - start_time
    #
    # ~~~ Method 2: Conversion to pandas DataFrame and then to NumPy array
    start_time = time.time()
    df = pd.read_parquet("C:\\Users\\thoma\\AppData\\Local\\Programs\\Python\\Python310\\Custom\\brains\\data\\train_spectrograms\\353733.parquet")
    numpy_array_pandas = df.values
    pandas_conversion_time = time.time() - start_time
    #
    # ~~~ Method 3: Load .npy files that have been created offline in advance
    start_time = time.time()
    numpy_array = np.load("C:\\Users\\thoma\\AppData\\Local\\Programs\\Python\\Python310\\Custom\\brains\\data\\spectrograms_npy\\train_spectrograms\\353733.npy")
    numpy_loading_time = time.time() - start_time
    #
    # ~~~ State the results
    print(f"Time taken for direct conversion:         {direct_conversion_time:.4}")
    print(f"Time taken for conversion through pandas: {pandas_conversion_time:.4}")
    print(f"Time taken for loading the .npy:          {numpy_loading_time:.4}")
else:
    #
    # ~~~ Method 1: Direct conversion to NumPy array
    for path in tqdm(spec_paths):
        numpy_array_direct = np.array(list(pq.read_table(path).to_pydict().values()), dtype=float)
    #
    # ~~~ Method 2: Conversion to pandas DataFrame and then to NumPy array
    for path in tqdm(spec_paths):
            numpy_array_pandas = pd.read_parquet("C:\\Users\\thoma\\AppData\\Local\\Programs\\Python\\Python310\\Custom\\brains\\data\\train_spectrograms\\353733.parquet").values
    #
    # ~~~ Method 3: Load .npy files that have been created offline in advance
    for path in tqdm(spec_npy_paths):
        numpy_array = np.load("C:\\Users\\thoma\\AppData\\Local\\Programs\\Python\\Python310\\Custom\\brains\\data\\spectrograms_npy\\train_spectrograms\\353733.npy")

#