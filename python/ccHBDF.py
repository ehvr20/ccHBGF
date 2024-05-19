import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix, hstack, diags, linalg

def cc_hbgf(clustering_matrix: np.array, verbose=False) -> np.array:
    """
    Perform consensus clustering using Hybrid Bipartite Graph Formulation (HBGF).

    This function performs the following steps:
    1. Calculates the optimal number of clusters `optimal_k` based on the input `clustering_matrix`.
    2. Constructs the adjacency matrix `A` of the hypergraph from the clustering matrix.
    3. Computes the normalized Laplacian matrix `L` from the adjacency matrix `A`.
    4. Performs Singular Value Decomposition (SVD) on the Laplacian matrix `L` using scipy.sparse.lingalg.svds.
    5. Normalizes the resulting singular vectors `U` and `V`.
    6. Uses KMeans to fit the normalized singular vectors for partitioning.
    7. Predicts the final consensus clustering labels.

    Parameters:
    clustering_matrix (numpy.ndarray): A 2D array where each column represents a clustering solution, 
                                       and each row represents an element being clustered.

    Returns:
    numpy.ndarray: A 1D array of consensus clustering labels for the elements.

    Example:
    >>> clustering_matrix = np.array([[1, 2, 1], [1, 2, 2], [2, 1, 1]])
    >>> labels = cc_hbgf(clustering_matrix)
    >>> print(labels)
    [0 1 0]
    """
    
    optimal_k = int(np.max(np.apply_along_axis(lambda x: np.unique(x).__len__(), 0, clustering_matrix)))
    
    if verbose:
        from datetime import datetime
        log_time = lambda: datetime.now().strftime('[%H:%M:%S] - ')
        print(f'{log_time()}Expected Clusters: {optimal_k}')

    # Construct Adjacency Matrix of Hypergraph
    solution_binary_matrices = []
    for solution in clustering_matrix.T:
        clusters = np.unique(solution)
        binary_matrix = (solution[:, np.newaxis] == clusters).astype(bool)
        solution_binary_matrices.append(csc_matrix(binary_matrix, dtype=bool))

    A = hstack(solution_binary_matrices, dtype=bool)
    del solution_binary_matrices
    
    if verbose:
        print(f'{log_time()}Computed A {A.shape}')

    # Calculated Laplacian Matrix
    D = diags(np.sqrt(A.sum(axis=0)).A1).tocsc()
    L = A.dot(linalg.inv(D))
    del A, D
    
    if verbose:
        print(f'{log_time()}Computed L {L.shape}')

    # Perform Singular Value Decomposition (SVD)
    singular_values = linalg.svds(L, k=optimal_k)
    del L
    
    if verbose:
        print(f'{log_time()}Computed SVDs')

    # Define left(U) and right(V) Singular Values
    U = singular_values[0]
    V = singular_values[2]
    del singular_values

    # Normalize to Unit Vector representations
    U = U/np.repeat(np.sqrt((U**2).sum(axis=1))[:, None], optimal_k, axis=1)
    V = V/np.repeat(np.sqrt((V**2).sum(axis=0))[None, :], optimal_k, axis=0)

    # Merge Bipartite Graph
    UVt = np.vstack([U, V.T])
    del U, V
    
    if verbose:
        print(f'{log_time()}Normalized U and V')

    # KMeans partitioning of Bipartite Graph
    kmeans = KMeans(n_clusters=optimal_k, random_state=0, init='k-means++')
    kmeans.fit(UVt)
    
    if verbose:
        print(f'{log_time()}KMeans Model Fitted')

    # Cluster Elements
    consensus_labels = kmeans.predict(UVt[:clustering_matrix.shape[0]])
    
    if verbose:
        print(f'{log_time()}Consensus Labels Predicted')

    return consensus_labels
