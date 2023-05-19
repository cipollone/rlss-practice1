# RLSS practical session 1

## Create virtual environment
Move to the local directory you would like to work from
Open terminal
Run "python -m venv [name of folder to install environment]

## Install
This package can be installed as usual:

    pip install .

Or, we can install a specific tested version of this package and its dependencies with:

    poetry install --no-dev

Omit the `--no-dev` option if you're installing for local development.

## Run
If installed with poetry, you can run the main function with:

    poetry run python -m <package-name>

or specific scripts with:

    poetry run python scripts/<python-file>
