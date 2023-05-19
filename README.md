# Python template

A template for starting Python 3 projects.

## Use

When you use this repo as template, you have to chage:

- This README
- The LICENSE
- Package name and info
- Add dependencies
- Personalize editor infos (e.g. python path in .vim/)


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
