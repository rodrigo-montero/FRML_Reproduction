# FRML Reproduction Project

Reproduction project for *Temporal and Spatial Reservoir Ensembling Techniques for Liquid State Machines*.

## Setup

Create a Python 3.10 virtual environment:

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data

Datasets are stored locally in:

```data/raw/```

The dataset folders are ignored by Git and should not be committed.

To download datasets:

```bash
python src/datasets/download_data.py
```

Currently downloaded:
N-MNIST
SHD
The DVSGesture dataset has failed to download for me. If that is the case go to 
```https://zenodo.org/records/8060604``` and download the two files manually. They are chunky ~3GB in total


## Experiments

the experiments are located in the experiments directory. Currently, they should be setup to run the correct models with
the correct parameters (we need to check) but without the whole dataset. 

TODO: Run them with the whole dataset to see if we get the expected results.
Extra: Run them with multiple seeds, but the paper I don't think did that.

Note: the experiments are named small but we should rename that eventually.

Experiments:
(same order as the table in the paper)
1) ``` run_nmnist_tepre_small.py ```
2) ``` run_nmnist_mulre_small.py ```
3) ``` run_shd_tepre_small.py ```
4) ``` run_dvsgesture_lsm_standard_small.py ```
5) ``` run_dvsgesture_lsm_receptive_small.py ```
6) The rest are for inspecting the data


## Models

we have:
1) lsm_paper.py
2) tepre.py
3) mulre.py
4) lsm.py (this one was the first one made but I don't think it is used)