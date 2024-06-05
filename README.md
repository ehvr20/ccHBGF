# Consensus Clustering with Hybrid Bipartite Graph Formulation (HBGF)

This repository provides an implementation of the `ccHBGF` function, which performs consensus clustering using Hybrid Bipartite Graph Formulation (HBGF).

## Overview

The `cc_hbgf` function performs consensus clustering by following these steps:
1. Construction of a `bipartite graph`
2. Definition of an Adjaceny Matrix `A`
3. Partitioning of the graph utilising either:
    - `Spectral Clustering`
    - `METIS`

## Installation

To use this function, you need to have Python installed along with the following packages:
- `numpy`
- `scipy`
- `sklearn`
- `pymetis` (optional)

You can install these packages using pip:

```bash
pip install numpy scipy scikit-learn pymetis
