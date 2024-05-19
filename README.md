# Consensus Clustering with Hybrid Bipartite Graph Formulation (HBGF)

This repository provides an implementation of the `cc_hbgf` function, which performs consensus clustering using Hybrid Bipartite Graph Formulation (HBGF).

## Overview

The `cc_hbgf` function performs consensus clustering by following these steps:
1. Calculates the optimal number of clusters `optimal_k` based on the input `clustering_matrix`.
2. Constructs the adjacency matrix `A` of the hypergraph from the clustering matrix.
3. Computes the normalized Laplacian matrix `L` from the adjacency matrix `A`.
4. Performs Singular Value Decomposition (SVD) on the Laplacian matrix `L` using `scipy.sparse.linalg.svds`.
5. Normalizes the resulting singular vectors `U` and `V`.
6. Uses KMeans to fit the normalized singular vectors for partitioning.
7. Predicts the final consensus clustering labels.

## Installation

To use this function, you need to have Python installed along with the following packages:
- `numpy`
- `scipy`
- `sklearn`

You can install these packages using pip:

```bash
pip install numpy scipy scikit-learn
