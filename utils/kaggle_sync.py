"""Tools for communicating with Kaggle. The main function of this file is to
publish the code in the "utils" folder as a Kaggle "utility script". Assuming
you have the Kaggle API set up on your computer, just run this file to do this.
"""
import tokenize
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

def strip_module_docstring(infile):
    """Get text of a file, minus an initial docstring (and initial comments)."""
    # from StackOverflow: https://stackoverflow.com/questions/156504/how-to-skip-the-docstring-using-regex

    insert_index = None
    f = open(infile)
    for tok, text, (srow, scol), (erow,ecol), l in tokenize.generate_tokens(f.readline):
        if tok == tokenize.COMMENT:
            continue
        elif tok == tokenize.STRING:
            insert_index = erow, ecol
            break
        else:
            break # No docstring found

    lines = open(infile).readlines()
    if insert_index is not None:
        erow = insert_index[0]
        return "".join(lines[erow:])
    else:
        return "".join(lines)
    
def all_utils_txt():
    """Collate the text of all Python files in the utils folder into a single
    string. Strips out module docstrings and internal imports.
    """
    all_files_txt = "### This file should not be edited directly. Run kaggle_sync.py to rebuild it."
    for path in Path('.').glob('*.py'):
        if path.name in ['__init__.py', 'utils.py', 'kaggle_sync.py']:
            continue

        this_file_txt = strip_module_docstring(path)
        # add separators
        this_file_txt = f"### From {path.name} ###\n" + this_file_txt + "\n\n\n"
        # strip internal imports
        this_file_txt = "\n".join(line for line in this_file_txt.split("\n")
                                  if not line.startswith("from ."))
        all_files_txt += this_file_txt
    return all_files_txt

def write_utils_file():
    """Write the text of all Python files in the utils folder to a single file
    called "utils.py".
    """
    with open("utils.py", "w") as fh:
        fh.write(all_utils_txt())
        
def push_to_kaggle():
    """Push "utils.py" to the utility script "allispaul/utils". Requires
    Kaggle API access (i.e., you need to have saved your API key to
    $HOME/.kaggle/kaggle.json.)"""
    api = KaggleApi()
    api.authenticate()
    api.kernels_push(".")
    
if __name__ == "__main__":
    write_utils_file()
    print("Wrote to utils.py.")
    try:
        push_to_kaggle()
        print("Pushed to allispaul/utils.")
    except OSError:
        print("Could not find kaggle.json. Script not pushed to Kaggle.")
    
    
        
