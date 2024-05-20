import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix, hstack, diags, linalg

def cc_hbgf(clustering_matrix: np.array, 
            k: int = None,
            tol: int = 1,
            random_state=None,
            method='spec',
            verbose: bool = False) -> np.array:
    """
    Perform consensus clustering using Hybrid Bipartite Graph Formulation (HBGF).

    This function performs consensus clustering on a `clustering_matrix`, which is a 2D array where each column 
    represents a clustering solution, and each row represents an element being clustered. It constructs a bipartite 
    graph with vertices representing the clusters and elements, and then partitions the hypergraph using spectral 
    clustering or METIS to generate final cluster labels.

    Parameters:
    clustering_matrix (numpy.ndarray): A 2D array where each column represents a clustering solution, 
                                       and each row represents an element being clustered.
    k (int, optional): The optimal number of clusters. If not provided, it's inferred from the input data.
    tol (int, optional): The tolerance for scipy.sparse.linalg.svds(). Where `0` is machine precision.
    random_state (int or RandomState instance, optional): Controls the randomness of the algorithm.
    method (str, optional): The method for graph partitioning. 'spec' for spectral clustering or 'metis' for METIS.
    verbose (bool, optional): Whether to print verbose output during processing.

    Returns:
    numpy.ndarray: A 1D array of consensus clustering labels for the elements.

    Example:
    >>> clustering_matrix = np.array([[1, 2, 1, 2], 
                                      [1, 2, 2, 1], 
                                      [1, 2, 1, 1]])
    >>> cc_hbgf(clustering_matrix)
    array([0, 1, 0])
    """
    
    if not k:
        k = int(np.max(np.apply_along_axis(lambda x: np.unique(x).size, 0, clustering_matrix)))
    
    if verbose:
        import datetime
        log_time = lambda: datetime.datetime.now().strftime('[%H:%M:%S] - ')
        print(f'{log_time()}Expected Clusters: {k}')

    # Construct hypergraph Adjacency Matrix (A)
    solution_binary_matrices = []
    for solution in clustering_matrix.T:
        clusters = np.unique(solution)
        binary_matrix = (solution[:, np.newaxis] == clusters).astype(bool)
        solution_binary_matrices.append(csc_matrix(binary_matrix, dtype=bool))

    A = hstack(solution_binary_matrices, dtype=bool)
    
    if verbose:
        print(f'{log_time()}Computed A {A.shape}')
        
    if method == 'spec':
        # Calculate Laplacian Matrix (L) from A
        D = diags(np.sqrt(A.sum(axis=0)).A1).tocsc()
        L = A.dot(linalg.inv(D))
        
        if verbose:
            print(f'{log_time()}Computed L {L.shape}')

        # Perform Singular Value Decomposition (SVD)
        U, _, V = linalg.svds(L, k, tol=tol, random_state=random_state)
        
        if verbose:
            print(f'{log_time()}Computed SVs')

        # Normalize left (U) and right (V) SVs to unit vectors
        U = U / np.linalg.norm(U, axis=1, keepdims=True)
        V = V / np.linalg.norm(V, axis=0, keepdims=True)

        # Merge SV representation of Bipartite Graph
        UVt = np.vstack([U, V.T])
        
        if verbose:
            print(f'{log_time()}Normalized U and V')

        # KMeans partitioning of Bipartite Graph
        kmeans = KMeans(n_clusters=k, random_state=random_state, init='k-means++')
        kmeans.fit(UVt)
        
        if verbose:
            print(f'{log_time()}KMeans Model Fitted')

        # Cluster Elements
        consensus_labels = kmeans.predict(UVt[:clustering_matrix.shape[0]])

    elif method == 'metis':
        from pymetis import part_graph, Options
        from scipy.sparse import bmat
        
        # Calculate W
        rowA, colA = A.shape
        W = bmat([[csc_matrix((colA, colA)), A.T],
                  [A, csc_matrix((rowA, rowA))]], 
                 format='csr', dtype=bool)
        
        if verbose:
            print(f'{log_time()}Computed W {W.shape}')
        
        # Convert Adjacency Matrix to pyMETIS format
        xadj, adjncy = W.indptr, W.indices
            
        if verbose:
            print(f'{log_time()}Converted W to pyMETIS format')
        
        # Define METIS options
        opt = Options()
        
        if random_state is not None:
            opt.seed = int(random_state)
            
        # Run METIS partitioning
        membership = part_graph(k, xadj=xadj, adjncy=adjncy, options=opt)[1]
        consensus_labels = np.array(membership)[colA:]

    else:
        raise ValueError('Invalid Graph Partitioning Method')

    if verbose:
        print(f'{log_time()}Consensus Labels Found')

    return consensus_labels

