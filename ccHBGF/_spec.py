"Copyright (C) 2024 E. H. von Rein"

import logging
from typing import Literal, Union, Optional

import numpy as np
from numpy.typing import NDArray

from sklearn.cluster import KMeans, kmeans_plusplus
from scipy.sparse import csc_matrix, hstack, diags, linalg

logger = logging.getLogger('ccHBGF')

def ccHBGF(clustering_matrix: np.ndarray,
		   n_clusters: Optional[int] = None,
		   tol: float = 0.1,
		   init: Literal['orthogonal', 'kmeans++'] = 'kmeans++',
		   random_state: Optional[Union[int, np.random.RandomState, np.random.Generator]] = None,
		   verbose: bool = False
		   ) -> np.ndarray:
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

def _construct_adj_matrix(matrix: np.ndarray
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

def _spectral_partitioning(adj: np.ndarray,
						   k: int,
						   tol: float = 0.1,
						   init: Literal['orthogonal', 'kmeans++'] = 'kmeans++',
						   random_state: Optional[Union[int, np.random.RandomState, np.random.Generator]] = None
						   ) -> np.ndarray:
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
		centers, _ = kmeans_plusplus(U, n_clusters=k, random_state=0)
	elif init == 'orthogonal':
		centers = _orthogonal_centers(U, n_clusters=k, random_state=random_state)

	logger.info('Initialized Centers')

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

def _orthogonal_centers(arr: NDArray,
						n_clusters: int,
						random_state: Optional[Union[int, np.random.RandomState, np.random.Generator]] = None
						) -> np.ndarray:
	"""
	Initialize cluster centers by selecting initial centers that are orthogonal to each other in the feature space.

	Parameters
	----------
	arr : ndarray
		Input array with data points.
	n_clusters : int
		Number of clusters.
	random_state : {None, int, numpy.random.Generator, numpy.random.RandomState}, optional
		Random state for reproducibility. Default is None.

	Returns
	-------
	ndarray
		Initialized cluster centers.
	"""

	n = arr.shape[0]

	rng = _parse_random_state(random_state)

	init = rng.choice(n)
	centers = arr[init, None]
	c = np.zeros(n)
	c[init] = 2 * n_clusters

	for j in range(1, n_clusters):
		c += np.abs(np.dot(arr, centers[j-1, :].T))
		m = np.argmin(c)
		centers = np.vstack((centers, arr[m, :]))
		c[m] = 2 * n_clusters

	return centers

def _parse_random_state(random_state: Union[None, int, np.random.RandomState, np.random.Generator]
						) -> {None, np.random.Generator, np.random.RandomState}:
	"""
	Pseudorandom number generator state used to generate resamples.

	If `random_state` is ``None`` (or `np.random`), the `np.random.RandomState` singleton is used.
	If `random_state` is an int, a new ``RandomState`` instance is used, seeded with `random_state`.
	If `random_state` is already a ``Generator`` or ``RandomState`` instance then that instance is used.
	
	Parameters
	----------
	arr : `ndarray`
		Input array.
	n_clusters : `int`
		Number of clusters.
	random_state : {None, int, `numpy.random.Generator`,
					`np.random.RandomState`}, optional
		Random seed for reproducibility.

	Returns
	-------
	{`np.random.RandomState`, `np.random.Generator`}
		Numpy random number generator instance.

	"""
	if random_state is None:
		return np.random.RandomState()
	elif isinstance(random_state, int):
		return np.random.RandomState(seed=random_state)
	elif isinstance(random_state, np.random.RandomState) or isinstance(random_state, np.random.Generator):
		return random_state
	else:
		raise ValueError("random_state be None, an int, or a numpy.random.RandomState instance")
	