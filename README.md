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
DVSGesture partially downloaded, but extraction may fail through Tonic