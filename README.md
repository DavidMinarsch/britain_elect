# Britain Elect

Sample project for MRP.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

### Prerequisites

This assumes you manage python versions with `pyenv` as outlined [here](https://github.com/pyenv/pyenv) and virtual environments with `pyenv-virtualenv` as outlined [here](https://github.com/pyenv/pyenv-virtualenv).

See `.python-version` for the python version required.

See `requirements.txt` for the libraries required.

### Setup Virtual Environment:

Manage python versions with `pyenv`. Check python versions installed include version specified in 
`.python-version` (i.e. :
```
$ pyenv versions
```

Manage virtual environments with `pyenv-virtualenvs`. Create a virtual environment:
```
$ pyenv virtualenv 3.7.2 britain_elect
```

### Install Dependencies:

CD into folder to activate the virtual environment and run:
```
$ pip install requirements.txt
```

## Or run from Command Line
```
$ python election15_britain.py
```
