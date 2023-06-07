# RLSS practical session 1

This repository contains [Gymnasium](https://github.com/Farama-Foundation/Gymnasium)
environments that we will use during the first practical session.
This is a configured [minigrid](https://github.com/Farama-Foundation/MiniGrid) environment with explicit transition and reward functions.


## Use

Install this package with pip as:

    pip install git+https://github.com/cipollone/rlss-practice1.git

Then, import the package and instantiate your environment of choice as:

    from rlss_practice1.environments import Room
    env = Room(seed=seed, failure=0.1, size=5)

For further documentation, see each environment docstring with `help()`.

## Development

Make sure to have a working Python 3.9 or 3.10 installation by running `python --version`.
These versions is available in most Linux distributions via the default package manager.
If not availabe, use [pyenv](https://github.com/pyenv/pyenv), then run from this directory: `pyenv local 3.9`. From now on, we assume `python` is `python3`.

This package is configured with [Poetry](https://python-poetry.org/). Poetry is a tool for developing Python packages. If this is installed in your system, running `poetry install` from the current directory will create a separate virtual environment and install all dependencies. In this case, you may proceed to section *Running*.

The other alternative is to manually create a virtual environment and install the dependencies within.
Create and activate a virtual environment, then install the package as:

    python -m venv <path>
    source <path>/bin/activate
    pip install .

To run parts of this package, it is necessary to enter this virtual environment for each new shell.
Depending on the installation method, this is `poetry shell` or `source <path>/bin/activate`.
