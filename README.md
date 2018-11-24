
# Trash recognition


## Installation

It is highly recommended (and assumed throughout this readme) that you create and activate a virtual environment before you start:

```shell
python3 -m venv env
source env/bin/activate
```

To exit the virtual environment
```shell 
deactivate
```

To install all dependencies (after activating the virtual environment):
```shell
pip install -r requirements.txt
```


## Deployment


## Running

### Data import

To run the default data import run: 
```shell
source env/bin/activate

python run_data_import.py.
```

To see any changable parameters for data import see defaults.json

### Learning and Analysis

First run data import described above, and keeping that virtual environment active run:

Then run  
```shell
python run_simple.py.
```

The command will run simple learning algorithms and rudimentary analyses.



