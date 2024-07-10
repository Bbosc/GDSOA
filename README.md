# Geometric Dynamical System for obstacle avoidance in manipulator configuration space

This project is a part of my master thesis @ LASA, EPFL.


The goal of this project is to develop a full-body obstacle avoidance based on dynamical system.
The dynamical system uses differential geometry concepts, specifically geodesics to avoid obstacles.
This DS evolves in an embedded representation of the configuration space, enhanced by a collision probability given by the Gaussian Mixture Mdoel (GMM) representation of the manipulator.

It can achieve a computational frequency > 200Hz for a 7-dimensional configuration space, on a i7-1360P CPU with 16 cores, and successfully achieves multi-obstacle avoidance in 
different scenarios.

## Installation

Clone the repository:

```bash
git clone git@github.com:Bbosc/GDSOA.git
```

### Install the dependencies

With pip:

```bash
python3 -m venv gdsoa
. ./gdsoa/bin/activate
pip install -r requirements.txt
```

With mamba/conda:

```bash
mamba env create -f environment.yml
```
### Test your installation

```bash
mamba activate gdsoa # or . ./gdsoa/bin/activate

python installation_test.py
```

## Test the algorithms

Several examples are available the **./examples** folder.


