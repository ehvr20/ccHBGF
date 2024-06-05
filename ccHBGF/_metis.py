"Copyright (C) 2024 E. H. von Rein"

import logging
from typing import Literal
import os
import subprocess
import tempfile

import numpy as np
from numpy import ndarray
from scipy.sparse import csc_matrix, hstack, block_array

logger = logging.getLogger('ccHBGF')

def ccHBGF(clustering_matrix: ndarray,
           n_clusters: int=None,
           temp_dir: str='./',
           seed: int=None,
           verbose: bool=False,
           **metis_options) -> np.array:
    """
    Perform consensus clustering using Hybrid Bipartite Graph Formulation (HBGF).

    This function performs consensus clustering of a `clustering_matrix`, which is a 2D array where each column
    represents a clustering solution, and each row represents an element being clustered. It constructs a bipartite
    graph with vertices representing the clusters and elements, and then partitions the hypergraph using METIS 
    partitioning to generate final cluster labels.

    Parameters
    ----------
    clustering_matrix : ndarray
        A 2D array where each column represents a clustering solution, and each row represents an element being clustered.

    n_clusters : int, optional
        Number of clusters. If not provided, the function automatically detects the number of clusters.

    temp_dir : str, optional
        Directory to store temporary files.
        
    seed : int, optional
        Controls the randomness of the algorithm.

    verbose : bool, optional
        Whether to print verbose output during processing.

    metis_options : dict, optional
        Additional kwargs to pass to METIS options.
        For more information try 'gpmetis -h'.

    Returns
    -------
    ndarray
        A 1D array of consensus clustering labels for the elements.
    """
    
    # Set verbosity level
    if verbose:
        logger.setLevel(logging.INFO)

    # Define expected number of clusters, if not given
    if not n_clusters:
        n_clusters = int(np.max(np.apply_along_axis(lambda x: np.unique(x).size, 0, clustering_matrix)))

    logger.info(f'Detected {n_clusters} clusters.')

    if n_clusters > 500:
        logger.warning(f'Large numbers of clusters detected. This may take a while.')

    # Construct hypergraph adjacency matrix (A)
    A = _construct_adj_matrix(clustering_matrix)

    logger.info(f'Hypergraph adjacency matrix (A) constructed with shape {A.shape}')

    # Partition Graph
    cluster_labels = _metis_partitioning(A, n_clusters, temp_dir, seed=seed, **metis_options)

    logger.info('Consensus Labels Found')

    return cluster_labels

def _construct_adj_matrix(matrix: ndarray) -> csc_matrix:
    """
    Construct a sparse adjacency matrix from a clustering matrix.

    This function constructs a sparse adjacency matrix from a clustering matrix, where each column represents
    a clustering solution, and each row represents an element being clustered. It converts each clustering
    solution into a binary matrix and concatenates them horizontally to form the adjacency matrix.

    Parameters
    ----------
    matrix : ndarray
        A 2D array where each column represents a clustering solution, and each row represents an element being clustered.

    Returns
    -------
    csc_matrix
        A sparse adjacency matrix in Compressed Sparse Column (CSC) format.
    """

    binary_matrices = []
    for solution in matrix.T:
        clusters = np.unique(solution)
        binary_matrix = (solution[:, np.newaxis] == clusters).astype(bool)
        binary_matrices.append(csc_matrix(binary_matrix, dtype=bool))

    return hstack(binary_matrices, format='csc', dtype=bool)

def _metis_partitioning(adj: ndarray,
                        n_parts: int,
                        temp_dir: dict = './',
                        **metis_options) -> ndarray:
    """
    Perform graph partitioning using the METIS library - gpmetis program.

    This function converts a bipartite graph represented by an adjacency matrix into a symmetric general graph
    adjacency matrix and performs graph partitioning using the METIS library.

    Parameters
    ----------
    adj : ndarray
        Adjacency matrix of the bipartite graph.
    n_parts : int
        Number of clusters (partitions).
    temp_dir : str
        Directory to store temporary files.
    metis_options : dict
        Additional options to pass to the METIS library.

    Returns
    -------
    ndarray
        Cluster membership labels.
    """

    # Convert Biparite Graph A to General Graph W adjacency matrix
    c = adj.shape[1]
    W = block_array([[None, adj.T],
                     [adj,  None ]],
                     format='csr', dtype=bool)

    logger.info(f'Encoded Biparite Graph adjacency matrix A to General Graph W adjacency matrix')

    # Convert Adjacency Matrix to pyMETIS format
    xadj, adjncy = W.indptr, W.indices

    logger.info('Converted W to METIS format')

    default_options = {
        'ptype': 'kway',
        'objtype': 'cut',
        'ncuts': 10,
        'minconn': True,
    }
    default_options.update(metis_options)

    options = _dict_to_flags(default_options)

    fmt_options = "\n\t".join(options)
    logger.info(f'Metis Options:\n\t{fmt_options}')

    with tempfile.TemporaryDirectory(dir=temp_dir) as temp_dir:
        num_nodes = int(len(xadj) - 1)
        num_edges = int(len(adjncy)/2)

        graph_file = os.path.join(temp_dir,'tmp.graph')

        with open(graph_file, 'w') as f:
            f.write(f"{num_nodes} {num_edges}\n")
            for i in range(num_nodes):
                neighbors = adjncy[xadj[i]:xadj[i+1]]
                neighbors_str = ' '.join(map(lambda x: str(x + 1), neighbors)) # adjust 0 index
                f.write(f"{neighbors_str}\n")

        logger.info(f'Temporary graph file generated: {graph_file}')

        # Define the gpmetis command with all the parameters
        command = ["gpmetis"] + options + [graph_file, str(n_parts)]

        # Run the command using subprocess
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        if result.returncode != 0:
            raise ChildProcessError(f"Error running gpmetis: {result.stderr.decode('utf-8')}")

        output_file = f'{graph_file}.part.{n_parts}'

        with open(output_file) as fp:
            membership = list(label.strip() for label in fp)


    return np.array(membership)[c:]

def _dict_to_flags(options) -> list:
    """Convert dictionary of options to METIS command-line flags."""
    flags = []
    for key, value in options.items():
        if isinstance(value, bool):
            if value:
                flags.append(f'-{key}')
        else:
            flags.append(f'-{key}={value}')
    return flags