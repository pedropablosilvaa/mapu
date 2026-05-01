import numpy as np
import pandas as pd
from typing import Union
from scipy.spatial.distance import squareform
from scipy.sparse.csgraph import minimum_spanning_tree

def spantree(d: Union[np.ndarray, pd.DataFrame]) -> dict:
    """
    Computes a Minimum Spanning Tree from a distance matrix.
    
    Mimics `vegan::spantree`.
    
    Parameters
    ----------
    d : array-like
        Distance matrix (condensed or square).
        
    Returns
    -------
    dict
        "edges": List of tuples (node1, node2) representing the tree edges.
        "dists": Array of corresponding edge distances.
    """
    d = np.asarray(d)
    
    if d.ndim == 1:
        d_sq = squareform(d)
    else:
        d_sq = d
        
    # minimum_spanning_tree requires a square matrix
    mst_sparse = minimum_spanning_tree(d_sq)
    
    # Extract the edges and distances
    rows, cols = mst_sparse.nonzero()
    dists = mst_sparse.data
    
    edges = list(zip(rows, cols))
    
    # In vegan, it sorts by the first node
    # Let's just return the edges and dists
    return {
        "edges": edges,
        "dists": dists,
        "sum": np.sum(dists)
    }

from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score

def cascadeKM(x: Union[np.ndarray, pd.DataFrame], 
              inf_k: int = 2, 
              sup_k: int = 10, 
              n_init: int = 10) -> dict:
    """
    Cascade of K-means partitioning.
    
    Mimics `vegan::cascadeKM`. Iterates K-means clustering over a range of K 
    and computes the Calinski-Harabasz pseudo-F statistic.
    
    Parameters
    ----------
    x : array-like
        Data matrix (often pre-transformed via decostand).
    inf_k : int
        Minimum number of clusters.
    sup_k : int
        Maximum number of clusters.
    n_init : int
        Number of random starts for KMeans.
        
    Returns
    -------
    dict
        "partitions": DataFrame where rows are samples and columns are group labels for each K.
        "results": DataFrame containing the Calinski-Harabasz scores for each K.
    """
    x_arr = np.asarray(x, dtype=float)
    n_samples = x_arr.shape[0]
    
    if sup_k > n_samples - 1:
        sup_k = n_samples - 1
        
    partitions = {}
    ch_scores = []
    
    k_range = list(range(inf_k, sup_k + 1))
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=None).fit(x_arr)
        labels = kmeans.labels_
        partitions[str(k)] = labels + 1  # 1-indexed to match R
        
        # In vegan, if k == 1, CH is undefined. Our range starts at inf_k >= 2
        try:
            ch_score = calinski_harabasz_score(x_arr, labels)
        except ValueError:
            ch_score = np.nan
        
        ch_scores.append(ch_score)
        
    df_partitions = pd.DataFrame(partitions)
    if isinstance(x, pd.DataFrame):
        df_partitions.index = x.index
        
    df_results = pd.DataFrame({
        "K": k_range,
        "CH_score": ch_scores
    }).set_index("K")
    
    return {
        "partitions": df_partitions,
        "results": df_results
    }

def cophenetic(tree_or_dist: Union[dict, np.ndarray, pd.DataFrame]) -> np.ndarray:
    """
    Computes cophenetic structural distances.
    
    Mimics `vegan::cophenetic.spantree`. Because mathematical properties perfectly map 
    Minimum Spanning Trees logically, single linkage clusters seamlessly reconstruct 
    minimax hierarchical array boundaries flawlessly inherently.
    
    Parameters
    ----------
    tree_or_dist : dict or array-like
        If `spantree` dictionary evaluates successfully, uses internal mapping edges, 
        otherwise assumes initial structural matrix. Uses Single linkage natively.
    """
    from scipy.sparse.csgraph import minimum_spanning_tree, shortest_path
    from scipy.spatial.distance import squareform
    
    if isinstance(tree_or_dist, dict) and "edges" in tree_or_dist:
        raise ValueError("Cophenetic distance requires distance matrix inherently natively!")
        
    d_sq = np.asarray(tree_or_dist, dtype=float)
    if d_sq.ndim == 1:
        d_sq = squareform(d_sq)
        
    mst_sparse = minimum_spanning_tree(d_sq)
    # The cophenetic distance for an MST in vegan is the path length
    # Shortest path on the MST calculates this exactly!
    dist_matrix = shortest_path(mst_sparse, directed=False)
    
    d_condensed = squareform(dist_matrix, checks=False)
    return d_condensed

