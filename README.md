# RLSS practical session 1

Installation and usage instructions for the first practical session.


## Intallation

Make sure to have a working Python 3.9 or 3.10 installation by running `python --version`.
These versions is available in most Linux distributions via the default package manager.
If not availabe, use [pyenv](https://github.com/pyenv/pyenv), then run from this directory: `pyenv local 3.9`. From now on, we assume `python` is `python3`.

This package is configured with [Poetry](https://python-poetry.org/). Poetry is a tool for developing Python packages. If this is installed in your system, running `poetry install` from the current directory will create a separate virtual environment and install all dependencies. In this case, you may proceed to section *Running*.

The other alternative is to manually create a virtual environment and install the dependencies within.
Create and activate a virtual environment, then install the package as:

    python -m venv <path>
    source <path>/bin/activate
    pip install .


## Runing

Navigate to the 'main' folder and enter the virtual environment

with `poetry shell` by running:
    #how to enter environment with poetry shell.

or

with `source <path>/bin/activate` by running:
    python environment.py
    	
depending on the installation method.
