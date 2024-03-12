
### ~~~
## ~~~ Function that tests whether or not code is being executed in kaggle
### ~~~

import os

def this_is_running_in_kaggle():
    return 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

