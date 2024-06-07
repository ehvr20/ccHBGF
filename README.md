# ccHBGF: Consensus Clustering with Hybrid Bipartite Graph Formulation

A python consensus clustering function utilising Hybrid Bipartite Graph Formulation (HBGF). HBGF is a graph-based consensus multi-source clustering technique. This method constructs a bipartite graph with two types of vertices: observations and clusters from different clusteirng solutions. An edge exists only between an observation vertex and a cluster vertex, indicating the object's membership in that cluster. The graph is then partitioned using spectral partitioning to derive consensus labels for all observations.

<p align="center">
  <img src="https://raw.githubusercontent.com/ehvr20/ccHBGF/main/img/bipartite_graph.png" alt="Overview of Consensus Clustering Workflow"/>
</p>

## Overview

The `ccHBGF` function performs consensus clustering by following these steps:
1. Definition of a bipartite graph adjaceny matrix `A`
2. Decomposition of `A` into a spectral embedding `UVt`
3. Clustering of `UVt` into a consensus labels

## Installation

```bash
pip install ccHBGF
```

## Example Usage

```python
from ccHBGF import ccHBGF

consensus_labels = ccHBGF(solutions_matrix, init='orthogonal', tol=0.1, verbose=True, random_state=0)

```
Where the `solutions_matrix` is of shape (m,n):
- m = the number of observations
- n = the number of different clustering solutions.

<p align="center">
  <img src="https://raw.githubusercontent.com/ehvr20/ccHBGF/main/img/workflow.svg" alt="Overview of Consensus Clustering Workflow"/>
</p>
*See example.ipynb for more detailed example usage.

## References

[1] Hu, Tianming, et al. "A comparison of three graph partitioning based methods for consensus clustering." Rough Sets and Knowledge Technology: First International Conference, RSKT 2006, Chongquing, China, July 24-26, 2006. Proceedings 1. Springer Berlin Heidelberg, 2006.

[2] Fern, Xiaoli Zhang, and Carla E. Brodley. "Solving cluster ensemble problems by bipartite graph partitioning." Proceedings of the twenty-first international conference on Machine learning. 2004.

[3] Ng, Andrew, Michael Jordan, and Yair Weiss. "On spectral clustering: Analysis and an algorithm." Advances in neural information processing systems 14 (2001).
