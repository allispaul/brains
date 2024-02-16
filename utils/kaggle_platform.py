
### ~~~
## ~~~ Function that tests whether or not code is being executed in kaggle
### ~~~

import os

def this_is_running_in_kaggle():
    try:
        import kaggle
        import kaggle_secrets
        kaggle_packages_found = True
    except ImportError:
        kaggle_packages_found = False
    return ('kaggle' in os.getcwd() and kaggle_packages_found)

# written by Tom
