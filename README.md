# brains
Kaggle competition on classifying harmful brain activity.

## Instructions

### Syncing between GitHub and Kaggle
You first need to link your GitHub with your Kaggle account. You can do this in your Kaggle account settings.

To save a notebook from Kaggle to GitHub, on the Kaggle "File" menu, click "Link to GitHub". Then, when you save the notebook, it will ask you to find a repository. You should be able to get to "allispaul/brains" (though let me know if you have issues, since it's a private repo). You can then write a commit message and commit the notebook.

To load a notebook from GitHub into Kaggle, make a new notebook, go to "File -> Import Notebook", click on "GitHub", find the repository, and find the notebook.

### Setting up a virtual environment

_(Only matters if you want to run locally, i.e., not on Kaggle!)_

Work in Python projects often happens in a virtual environment: an isolated system containing a Python installation, packages and other software. This allows different and possibly conflicting versions of software to exist on the same computer, while ensuring that you're using the same version of everything as your collaborators.

To set up the virtual environment for this repository:
1. Install [conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).
2. `conda create --name brains --file environment.yml`. This will create a new virtual environment named `brains` (you can choose a different name if you prefer) and install the packages listed in the file `environment.yml`.
3. `conda activate brains`. To leave the virtual environment, use `conda deactivate`.
4. Assuming you're running Jupyter Notebook or Jupyter Lab, run `python -m ipykernel install --user --name=brains` with the environment active. This will add the environment to your list of kernels in Jupyter.

You can add more packages to the spec if you want: install them with conda or pip, then run `conda env export --from-history > environment.yml` to regenerate the spec. You should delete the line starting with `prefix:` (which will be specific to your computer) before pushing.

### Downloading the data

_(Again, only matters if you want to work locally.)_

The virtual environment includes the Kaggle Python package and associated command line interface, which you can [read about here](https://www.kaggle.com/docs/api). Follow the steps on that page to generate an API token and put it in the right place on your computer. Then run
```
kaggle config set -n competition -v hms-harmful-brain-activity-classification
```
to set this competition as your default competition, and
```
kaggle competitions download
```
(in your data folder) to download the data (18.4GB zip file). I would recommend doing this in a folder called `data/` inside the project folder (which Git will ignore).

