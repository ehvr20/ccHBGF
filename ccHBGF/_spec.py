import logging
from typing import Literal, Union, Optional

import numpy as np
from numpy.typing import NDArray

from sklearn.cluster import KMeans, kmeans_plusplus
from scipy.sparse import csc_matrix, hstack, diags, linalg

logger = logging.getLogger('ccHBGF')

def ccHBGF(clustering_matrix: NDArray,
           n_clusters: Optional[int] = None,
           tol: float = 0.1,
           init: Literal['orthogonal', 'kmeans++'] = 'kmeans++',
           random_state: Optional[Union[int, np.random.RandomState, np.random.Generator]] = None,
           verbose: bool = False
           ) -> NDArray:
    """
    Perform consensus clustering using Hybrid Bipartite Graph Formulation (HBGF).

    This function performs consensus clustering on a `clustering_matrix`, which is a 2D array where each column
    represents a clustering solution and each row represents an element being clustered. It constructs a bipartite
    graph with vertices representing the clusters and elements, and then partitions the graph using spectral
    partitioning to generate final cluster labels.

    Parameters
    ----------
    clustering_matrix : ndarray
        A 2D array where each column represents a clustering solution, and each row represents an element being clustered.

    n_clusters : int, optional
        The number of clusters. If not provided, the function automatically detects the number of clusters.

    tol : float, optional
        The tolerance for scipy.sparse.linalg.svds(), where `0` is machine precision.

    init : {'orthogonal', 'kmeans++'}, optional
        Method for initializing KMeans centers. Default is 'kmeans++'.

    random_state : {None, int, numpy.random.Generator, numpy.random.RandomState}, optional
        Controls the randomness of the algorithm for reproducibility. Default is None.

    verbose : bool, optional
        Whether to print verbose output during processing. Default is False.

    Returns
    -------
    ndarray
        A 1D array of consensus clustering labels for the elements.
    """

    # Check Input Parameters
    assert init in ['orthogonal', 'kmeans++'], f"No center initialization method: {init}.\nAvailable methods:\n\t- 'orthogonal'\n\t- 'kmeans++'"

    # Set verbosity level
    if verbose:
        logger.setLevel(logging.INFO)

    # Define expected number of clusters, if not given
    if not n_clusters:
        n_clusters = int(np.max(np.apply_along_axis(lambda x: np.unique(x).size, 0, clustering_matrix)))

    logger.info(f'Detected {n_clusters} clusters.')

    if n_clusters > 500:
        logger.warning(f'Large numbers of clusters detected. This may take a while.')

    # Construct graph adjacency matrix (A)
    A = _construct_adj_matrix(clustering_matrix)

    logger.info(f'Graph adjacency matrix (A) constructed with shape {A.shape}')

    # Derive cluster labels using spectral partitioning of graph
    cluster_labels = _spectral_partitioning(A, n_clusters, tol, init, random_state)

    logger.info('Consensus Labels Found')

    return cluster_labels

def _construct_adj_matrix(matrix: NDArray
                          ) -> csc_matrix:
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

def _spectral_partitioning(adj: NDArray,
                           k: int,
                           tol: float = 0.1,
                           init: Literal['orthogonal', 'kmeans++'] = 'kmeans++',
                           random_state: Optional[Union[int, np.random.RandomState, np.random.Generator]] = None
                           ) -> NDArray:
    """
    Perform spectral partitioning of a graph.

    This function performs spectral partitioning of a graph represented by an adjacency matrix.
    It calculates the Laplacian matrix from the adjacency matrix, performs Singular Value
    Decomposition (SVD) of the Laplacian, normalizes the left (U) and right (V) singular vectors to unit
    length, initializes cluster centers using the KMeans++ or orthogonal method, merges the
    singular vector representation of the bipartite graph, and partitions it using KMeans.

    Parameters
    ----------
    adj : ndarray
        Adjacency matrix of the graph.
    k : int
        Number of clusters.
    tol : float, optional
        Tolerance for Singular Value Decomposition (SVD). Default is 0.1.
    init : Literal['orthogonal', 'kmeans++'], optional
        Initialization method for cluster centers. Options are 'orthogonal' or 'kmeans++'. Default is 'kmeans++'.
    random_state : {None, int, numpy.random.Generator, numpy.random.RandomState}, optional
        Random state for reproducibility. Default is None.

    Returns
    -------
    ndarray
        Cluster membership labels.
    """

    # Calculate Laplacian Matrix (L) from A
    D = diags(np.sqrt(adj.sum(axis=0)).A1).tocsc()
    L = adj.dot(linalg.inv(D))

    logger.info(f'Transformed A to Laplacian Matrix (L) of shape {L.shape}')

    # Perform Singular Value Decomposition (SVD)
    U, _, V = linalg.svds(L, k, tol=tol, random_state=random_state)

    logger.info('Decomposed L into Singular Values (SVs)')

    # Normalize left (U) and right (V) SVs to unit vectors
    U = U / np.linalg.norm(U, axis=1, keepdims=True)
    V = V / np.linalg.norm(V, axis=0, keepdims=True)

    logger.info('Normalized SVs')

    if init == 'kmeans++':
        centers, _ = kmeans_plusplus(U, n_clusters=k, random_state=random_state)
        
        logger.info('Initialized Centers')
        
    elif init == 'orthogonal':
        centers, _ = kmeans_plusplus(U, n_clusters=k, random_state=random_state)
        
        logger.info('Initialized Centers')
        
        centers = _orthogonalize_centers(centers)
        
        logger.info('Orthogonalized Centers')


    # Merge SV representation of Bipartite Graph
    n = U.shape[0]
    UVt = np.vstack([U, V.T])

    # KMeans partitioning of Bipartite Graph
    kmeans = KMeans(n_clusters=k, init=centers, random_state=random_state)
    kmeans.fit(UVt)

    logger.info('KMeans model fitted to UVt')

    # Cluster Elements (U)
    membership = kmeans.predict(UVt[:n])

    return membership

def _orthogonalize_centers(vectors: NDArray
                           ) -> NDArray:
    """
    Perform Modified Gram-Schmidt (MGS) orthogonalization on a set of input vectors.

    MGS is a variant of the Gram-Schmidt process used to orthogonalize a set of vectors.
    Unlike the classical Gram-Schmidt process, MGS calculates all projections before subtracting them,
    which improves numerical stability and reduces round-off errors.

    Parameters
    ----------
    vectors : ndarray
        Input array of shape (m, n), where m is the number of samples (vectors)
        and n is the dimensionality of each vector. Each row represents a vector to be orthogonalized.

    Returns
    -------
    ndarray
        Orthogonalized vectors of the same shape as the input array.
        Each row of the returned array represents an orthogonal vector corresponding to the input vectors.

    Notes
    -----
    The Modified Gram-Schmidt algorithm works by iteratively orthogonalizing the input vectors.
    It subtracts the projections of each vector onto previously orthogonalized vectors.
    The resulting orthogonal vectors are normalized to unit length.
    """
    
    ortho_vectors = np.copy(vectors)
    for i in range(ortho_vectors.shape[0]):
        for j in range(i):
            projection = np.dot(ortho_vectors[i, :], ortho_vectors[j, :])
            ortho_vectors[i, :] -= projection * ortho_vectors[j, :]
        norm = np.linalg.norm(ortho_vectors[i, :])
        if norm != 0:
            ortho_vectors[i, :] /= norm
    return ortho_vectors
