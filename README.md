# brains
Kaggle competition on classifying harmful brain activity.

## Working on Kaggle
### Sharing notebooks between GitHub and Kaggle
You first need to link your GitHub with your Kaggle account. You can do this in your Kaggle account settings.

To save a notebook from Kaggle to GitHub, on the Kaggle "File" menu, click "Link to GitHub". Then, when you save the notebook, it will ask you to find a repository. You should be able to get to "allispaul/brains" (though let me know if you have issues, since it's a private repo). You can then write a commit message and commit the notebook.

Loading a notebook from GitHub into Kaggle does not seem to be working. So instead, download the notebook you want to use, make a new notebook in Kaggle, go to "File -> Import Notebook -> File", and upload the file.

### Using Python scripts on Kaggle
You can upload Python scripts to Kaggle as notebooks, designate them as "utility scripts", and attach them to your notebooks using "File -> Add utility script". Once you've done that, you can import them as you would a module.

You have write access to a script called `allispaul/utils`. This is auto-generated by pasting together the scripts in the utils folder; as a result, `from utils import X` should function identically for a user on Kaggle who's attached this script as for a local user who's running a notebook in the main project directory.

_(Note that `from utils.data_handling import X` will **not** function identically. I don't think we can replicate a full Python package structure with a Kaggle utility script. I would suggest we avoid this.)_

If you edit stuff in the utils folder, you should regenerate the script. To do this, run `python kaggle_sync.py" from within the utils folder. Pushing to Kaggle requires you to have Kaggle API access (see below) -- if you have issues with this, you can also copy-paste the contents of `utils.py` to `allispaul/utils`.

### API access
Go to "Settings -> API" on Kaggle and click "Create New Token". This will give you a file called "kaggle.json", which you should put in `C:\Users\<username>\.kaggle\` (on Windows) or `~/.kaggle/` (on Linux/Mac). You shouldn't need to do anything else. This will allow the Kaggle Python package (which you can get by `pip install kaggle` or `conda install kaggle`) and command line interface to work.

## Working locally
### Setting up a virtual environment

Work in Python projects often happens in a virtual environment: an isolated system containing a Python installation, packages and other software. This allows different and possibly conflicting versions of software to exist on the same computer, while ensuring that you're using the same version of everything as your collaborators.

To set up the virtual environment for this repository:
1. Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
2. Look at the `environment.yml` file in this repository. If you want to install pytorch without CUDA support (e.g. you don't have a GPU), delete the `pytorch-cuda` line.
3. `conda create --name brains --file environment.yml`. This will create a new virtual environment named `brains` (you can choose a different name if you prefer) and install the packages listed in the file `environment.yml`.
4. `conda activate brains`. To leave the virtual environment, use `conda deactivate`.
5. To check that CUDA is working, `python -c "import torch; print(torch.cuda.is_available())"` should give you `True`.
6. Assuming you're running Jupyter Notebook or Jupyter Lab, run `python -m ipykernel install --user --name=brains` with the environment active. This will add the environment to your list of kernels in Jupyter.

You can add more packages to the spec if you want: install them with conda or pip, then run `conda env export --from-history > environment.yml` to regenerate the spec. You should delete the line starting with `prefix:` (which will be specific to your computer) before pushing.

### Downloading the data

The virtual environment includes the Kaggle Python package and associated command line interface, which you can [read about here](https://www.kaggle.com/docs/api). Follow the steps under "API access" above to set up API access. Then run
```
kaggle config set -n competition -v hms-harmful-brain-activity-classification
```
to set this competition as your default competition, and
```
kaggle competitions download
```
(in your data folder) to download the data (18.4GB zip file). I would recommend doing this in a folder called `data/` inside the project folder (which Git will ignore).


## Running Tom's Code

In order to reproduce the file `simple_nn_on_spectrograms.py` you may just need to modify the import statements of Tom's custom stuff. Indeed these can depend upon your python environment. For your future use in other projects, I recommend that you directly refer to my (Tom's) standalone packaging of these features at
 - https://github.com/ThomasLastName/fit
 - https://github.com/ThomasLastName/quality_of_life

where I will be maintaining them, too.
