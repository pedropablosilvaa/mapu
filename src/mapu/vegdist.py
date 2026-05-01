import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from typing import Union

def vegdist(x: Union[np.ndarray, pd.DataFrame], 
            method: str = "bray", 
            binary: bool = False,
            diag: bool = False,
            upper: bool = False) -> np.ndarray:
    """
    Computes dissimilarity indices for community ecologists.
    
    This function mimics the behavior of `vegan::vegdist` in R.
    
    Parameters
    ----------
    x : array-like
        Community data matrix (rows are samples, columns are species).
    method : str, default "bray"
        Distance method. Available: "bray", "jaccard", "euclidean", "manhattan", "canberra", "cosine".
    binary : bool, default False
        If True, data are binarized into presence/absence (0/1) before calculating.
    diag : bool, default False
        Not directly used in Python's standard output, but provided for API similarity.
    upper : bool, default False
        If True, returns a square symmetric distance matrix. If False, returns 
        the condensed distance matrix (1D array like R's `dist`).
        
    Returns
    -------
    np.ndarray
        Distance matrix (condensed 1D array by default, or 2D square if upper=True).
    """
    x_arr = np.asarray(x, dtype=float)
    
    if binary:
        x_arr = (x_arr > 0).astype(float)
        
    if method == "bray":
        dist_condensed = pdist(x_arr, metric="braycurtis")
        
    elif method == "jaccard":
        if binary:
            dist_condensed = pdist(x_arr, metric="jaccard")
        else:
            # Quantitative Jaccard (Ruzicka)
            # In vegan: jaccard = 2B / (1+B) where B is Bray-Curtis
            bray = pdist(x_arr, metric="braycurtis")
            dist_condensed = (2 * bray) / (1 + bray)
            
    elif method == "euclidean":
        dist_condensed = pdist(x_arr, metric="euclidean")
        
    elif method == "manhattan":
        dist_condensed = pdist(x_arr, metric="cityblock")
        
    elif method == "canberra":
        # Scipy's canberra sum(|u-v|/(|u|+|v|)) correctly handles zeros.
        # However, R's vegan divides the sum by the number of features with non-zero
        # entries in the two rows to normalize it.
        dist_condensed = pdist(x_arr, metric="canberra")
        
        # Calculate number of non-zero entries per pair
        B = (x_arr == 0).astype(int)
        shared_zeros_sq = B @ B.T
        n_samples, n_features = x_arr.shape
        nz = n_features - shared_zeros_sq[np.triu_indices(n_samples, k=1)]
        # Avoid division by zero for completely empty pair matches
        nz[nz == 0] = 1
        
        dist_condensed = dist_condensed / nz
        
    elif method == "cosine":
        dist_condensed = pdist(x_arr, metric="cosine")
        
    elif method == "gower":
        # standardized by range
        ranges = np.ptp(x_arr, axis=0)
        ranges[ranges == 0] = 1.0 # avoid div by zero
        x_norm = x_arr / ranges
        dist_condensed = pdist(x_norm, metric="cityblock") / x_arr.shape[1]
        
    elif method == "mahalanobis":
        try:
            vi = np.linalg.inv(np.cov(x_arr, rowvar=False))
            dist_condensed = pdist(x_arr, metric="mahalanobis", VI=vi)
        except np.linalg.LinAlgError:
            raise ValueError("Covariance matrix is singular, cannot compute Mahalanobis distance.")
            
    elif method == "horn":
        # Morisita-Horn overlap index
        N = np.sum(x_arr, axis=1)
        N_sq = N**2
        N_sq[N_sq == 0] = 1.0
        lam = np.sum(x_arr**2, axis=1) / N_sq 
        
        cross = x_arr @ x_arr.T 
        lam_ij = lam[:, np.newaxis] + lam[np.newaxis, :]
        N_ij = N[:, np.newaxis] * N[np.newaxis, :]
        
        with np.errstate(divide='ignore', invalid='ignore'):
            sim = np.where((lam_ij * N_ij) > 0, (2.0 * cross) / (lam_ij * N_ij), 0)
            
        np.fill_diagonal(sim, 1.0)
        dist_condensed = squareform(1.0 - sim, checks=False)
        
    elif method == "kulczynski":
        x_sums = np.sum(x_arr, axis=1)
        n_sites = x_arr.shape[0]
        sim = np.zeros((n_sites, n_sites))
        
        for i in range(n_sites):
            for j in range(i, n_sites):
                mins = np.sum(np.minimum(x_arr[i], x_arr[j]))
                if x_sums[i] > 0 and x_sums[j] > 0:
                    val = 0.5 * (mins / x_sums[i] + mins / x_sums[j])
                else:
                    val = 0.0
                sim[i, j] = val
                sim[j, i] = val
                
        np.fill_diagonal(sim, 1.0)
        dist_condensed = squareform(1.0 - sim, checks=False)
        
    elif method == "chord":
        # Chord distance is structurally euclidean variants compiled over normalized dimensions natively
        row_norms = np.sqrt(np.sum(x_arr**2, axis=1, keepdims=True))
        with np.errstate(divide='ignore', invalid='ignore'):
            x_norm = np.where(row_norms == 0, 0, x_arr / row_norms)
        dist_condensed = pdist(x_norm, metric="euclidean")
        
    elif method == "gower":
        ranges = np.ptp(x_arr, axis=0)
        valid = ranges > 0
        p_valid = np.sum(valid)
        if p_valid == 0:
            dist_condensed = np.zeros(n * (n - 1) // 2)
        else:
            x_scaled = x_arr[:, valid] / ranges[valid]
            dist_condensed = pdist(x_scaled, metric="cityblock") / p_valid
            
    else:
        raise ValueError(f"Method '{method}' is not supported yet.")
            
    # Format into DataFrame if possible
    if upper or isinstance(x, pd.DataFrame):
        out_mat = squareform(dist_condensed)
        if isinstance(x, pd.DataFrame):
            return pd.DataFrame(out_mat, index=x.index, columns=x.index)
        return out_mat
        
    return dist_condensed

def designdist(x: Union[np.ndarray, pd.DataFrame], 
               method: str = "(A+B-2*J)/(A+B)", 
               terms: str = "quadratic") -> Union[np.ndarray, pd.DataFrame]:
    """
    Design your own distance metric.
    
    Mimics `vegan::designdist`. Evaluates distance metrics using formulaic 
    string commands parsing variables A, B, and J directly natively.
    
    Parameters
    ----------
    x : array-like
        Community data matrix.
    method : str
        Formula string combining elements A, B, and J.
    terms : str
        Term calculation logic: "binary", "quadratic", "minimum".
        
    Returns
    -------
    array-like
        Square distance matrix resulting from formula evaluations.
    """
    x_arr = np.asarray(x, dtype=float)
    n = x_arr.shape[0]
    
    A_mat = np.zeros((n, n))
    B_mat = np.zeros((n, n))
    J_mat = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if terms == "binary":
                i_vec = x_arr[i] > 0
                j_vec = x_arr[j] > 0
                A_mat[i, j] = np.sum(i_vec)
                B_mat[i, j] = np.sum(j_vec)
                J_mat[i, j] = np.sum(i_vec & j_vec)
            elif terms == "quadratic":
                A_mat[i, j] = np.sum(x_arr[i]**2)
                B_mat[i, j] = np.sum(x_arr[j]**2)
                J_mat[i, j] = np.sum(x_arr[i] * x_arr[j])
            elif terms == "minimum":
                A_mat[i, j] = np.sum(x_arr[i])
                B_mat[i, j] = np.sum(x_arr[j])
                J_mat[i, j] = np.sum(np.minimum(x_arr[i], x_arr[j]))
            else:
                raise ValueError("terms must be 'binary', 'quadratic', or 'minimum'")
                
    local_scope = {"A": A_mat, "B": B_mat, "J": J_mat, "np": np}
    out = eval(method, {"__builtins__": {}}, local_scope)
    
    if isinstance(x, pd.DataFrame):
        return pd.DataFrame(out, index=x.index, columns=x.index)
    return out

from scipy.sparse.csgraph import shortest_path

def stepacross(d: Union[np.ndarray, pd.DataFrame], 
               toolong: float = 1.0) -> Union[np.ndarray, pd.DataFrame]:
    """
    Shortest-path matrix distances (Stepacross).
    
    Mimics `vegan::stepacross`. Interrogates dissimilarities mathematically stripping 
    disconnections mapping sparse ceilings (toolong bounds), substituting arrays naturally 
    evaluating completely interconnected shortest paths topologies cleanly.
    
    Parameters
    ----------
    d : array-like
        Distance matrix arrays.
    toolong : float
        Threshold maximum substituting boundaries.
        
    Returns
    -------
    array-like
        Configured shortest-path arrays mapped linearly.
    """
    d_arr = np.asarray(d, dtype=float)
    is_df = isinstance(d, pd.DataFrame)
    
    if d_arr.ndim == 1:
        d_arr = squareform(d_arr)
        
    d_graph = d_arr.copy()
    d_graph[d_graph >= toolong] = np.inf
    np.fill_diagonal(d_graph, 0.0)
    
    out = shortest_path(d_graph, method='auto', directed=False)
    
    if is_df:
        return pd.DataFrame(out, index=d.index, columns=d.columns)
    return out
