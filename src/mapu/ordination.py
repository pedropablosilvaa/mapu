import numpy as np
import pandas as pd
from typing import Union, Tuple
from sklearn.manifold import MDS
from mapu.vegdist import vegdist
from scipy.spatial.distance import squareform, is_valid_y, is_valid_dm

def cmdscale(D: Union[np.ndarray, pd.DataFrame], 
             k: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classical Multidimensional Scaling (PCoA / cmdscale).
    
    This function mimics `stats::cmdscale` often used in R for PCoA.
    
    Parameters
    ----------
    D : array-like
        Distance matrix (can be square 2D or condensed 1D).
    k : int, default 2
        Number of dimensions.
        
    Returns
    -------
    points : np.ndarray
        Coordinates of the points.
    eig : np.ndarray
        Eigenvalues.
    """
    D = np.asarray(D)
    
    # If D is a condensed distance matrix, make it square
    if D.ndim == 1:
        D = squareform(D)
        
    # Double centering
    n = D.shape[0]
    # D^2
    D2 = D**2
    # Centering matrix: H = I - 1/n * J
    H = np.eye(n) - np.ones((n, n)) / n
    # B = -1/2 * H * D^2 * H
    B = -0.5 * H.dot(D2).dot(H)
    
    # Eigen decomposition (using eigh for symmetric matrices)
    eigvals, eigvecs = np.linalg.eigh(B)
    
    # Sort eigenvalues and eigenvectors in descending order
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    
    # Keep only top k positive eigenvalues
    w = np.where(eigvals > 0)[0]
    if len(w) < k:
        k = len(w)
        
    eigvals_k = eigvals[:k]
    eigvecs_k = eigvecs[:, :k]
    
    # Compute coordinates: vectors * sqrt(values)
    points = eigvecs_k * np.sqrt(eigvals_k)
    
    return points, eigvals

def metaMDS(x: Union[np.ndarray, pd.DataFrame], 
            distance: str = "bray", 
            k: int = 2, 
            n_init: int = 10,
            max_iter: int = 300) -> np.ndarray:
    """
    Non-metric Multidimensional Scaling (NMDS).
    
    A simplified equivalent of `vegan::metaMDS`.
    For true `metaMDS` equivalence, it should do Wisconsin/sqrt transformations,
    procrustes rotations, etc. This is a basic NMDS with multiple random starts.
    
    Parameters
    ----------
    x : array-like
        Community data matrix.
    distance : str, default "bray"
        Distance metric to use.
    k : int, default 2
        Number of dimensions.
    n_init : int, default 10
        Number of random initializations to find the global minimum stress.
    max_iter : int, default 300
        Maximum number of iterations for a single run.
        
    Returns
    -------
    np.ndarray
        Coordinates of the points in k dimensions.
    """
    # 1. Compute distance matrix
    dist_matrix = vegdist(x, method=distance, upper=True)
    
    # 2. Run Non-metric MDS using scikit-learn
    # We use precomputed distance and metric=False for non-metric MDS
    try:
        # scikit-learn >= 1.8 (or whenever metric_mds was introduced)
        mds = MDS(n_components=k, 
                  metric_mds=False,
                  init="random",
                  n_init=n_init, 
                  max_iter=max_iter, 
                  metric="precomputed",
                  random_state=None)  
    except TypeError:
        # scikit-learn < 1.8 (init might not be a valid keyword argument)
        mds = MDS(n_components=k, 
                  metric=False,
                  n_init=n_init, 
                  max_iter=max_iter, 
                  dissimilarity="precomputed",
                  random_state=None)  
    
    mds_result = mds.fit(dist_matrix)
    
    # In a full port, we would also return species scores, stress values, etc.
    # For now, return the sample coordinates
    return mds_result.embedding_

from scipy.spatial import procrustes as scipy_procrustes

def procrustes(X: np.ndarray, Y: np.ndarray) -> dict:
    """
    Procrustes Rotation.
    
    Mimics `vegan::procrustes(..., symmetric=TRUE)`.
    Rotates configuration Y to maximum similarity with X.
    
    Parameters
    ----------
    X : np.ndarray
        Target matrix.
    Y : np.ndarray
        Matrix to be rotated.
        
    Returns
    -------
    dict
        "m2": The symmetric Procrustes sum of squares.
        "X": Standarized target matrix.
        "Yrot": Rotated matrix Y.
    """
    X_std, Y_rot, disparity = scipy_procrustes(X, Y)
    return {
        "m2": disparity,
        "X": X_std,
        "Yrot": Y_rot
    }

def rda(x: Union[np.ndarray, pd.DataFrame], 
            Y: Union[np.ndarray, pd.DataFrame, None] = None) -> dict:
    """
    Redundancy Analysis (Constrained / Unconstrained Principal Component Analysis).
    
    Mimics `vegan::rda`. If `Y` is provided, performs constrained redundancy 
    analysis distributing inertias between fitted constraints and residuals.
    If `Y` is None, performs unconstrained PCA.
    
    Parameters
    ----------
    x : array-like
        Community data matrix (n_sites, n_species).
    Y : array-like, optional
        Environmental/predictor matrix (n_sites, n_variables).
        
    Returns
    -------
    dict
        Contains:
        - "tot.chi": Total variance (inertia).
        - "CCA": Dict of constrained inertias ("eigenvalues", "tot.chi") if Y provided.
        - "CA": Dict of unconstrained inertias ("eigenvalues", "tot.chi").
    """
    x_arr = np.asarray(x, dtype=float)
    n, p = x_arr.shape
    
    if n <= 1:
        raise ValueError("Cannot perform RDA on less than 2 samples.")
        
    # Center X
    x_c = x_arr - np.mean(x_arr, axis=0)
    
    res = {
        "tot.chi": 0.0,
        "CA": None,
        "CCA": None
    }
    
    if Y is not None:
        y_arr = np.asarray(Y, dtype=float)
        if y_arr.shape[0] != n:
            raise ValueError("Dimensions of constraint Y must match rows of community matrix x.")
            
        # Center Y
        y_c = y_arr - np.mean(y_arr, axis=0)
        
        # Multiple Linear Regression (X onto Y)
        # Using least squares to find Beta
        beta, _, _, _ = np.linalg.lstsq(y_c, x_c, rcond=None)
        
        # Fitted and Residuals
        x_fitted = y_c @ beta
        x_res = x_c - x_fitted
        
        # Helper inner function
        def extract_inertias(mat):
            mat_scaled = mat / np.sqrt(n - 1)
            # Standard SVD 
            _, S, _ = np.linalg.svd(mat_scaled, full_matrices=False)
            eig = S ** 2
            eig = eig[eig > 1e-12]
            return {"eigenvalues": eig, "tot.chi": np.sum(eig)}
            
        res["CCA"] = extract_inertias(x_fitted)
        res["CA"] = extract_inertias(x_res)
        res["tot.chi"] = res["CCA"]["tot.chi"] + res["CA"]["tot.chi"]
        
    else:
        # Unconstrained
        x_scale = x_c / np.sqrt(n - 1)
        U, S, Vt = np.linalg.svd(x_scale, full_matrices=False)
        eigenvalues = S ** 2
        eigenvalues = eigenvalues[eigenvalues > 1e-12]
        tot_chi = np.sum(eigenvalues)
        
        res["CA"] = {
            "eigenvalues": eigenvalues,
            "tot.chi": tot_chi
        }
        res["tot.chi"] = tot_chi
        
    return res

def envfit(X: np.ndarray, env: pd.DataFrame, permutations: int = 999) -> dict:
    """
    Fits Environmental Vectors onto an Ordination Grid.
    
    Mimics `vegan::envfit` for continuous variables.
    Calculates multiple regression of environmental vectors on ordination axes.
    
    Parameters
    ----------
    X : np.ndarray
        Ordination configuration/scores (n_sites, n_dimensions).
    env : pd.DataFrame
        Continuous environmental variables matrix (n_sites, n_variables).
    permutations : int
        Number of permutations to assess significance.
        
    Returns
    -------
    dict
        "vectors": Array of direction cosines (normalized regression coefficients).
        "r2": Array of multiple correlation coefficients (R-squared).
        "pvals": Permutation significance values.
    """
    X_arr = np.asarray(X, dtype=float)
    env_arr = np.asarray(env, dtype=float)
    
    n_sites, n_dims = X_arr.shape
    _, n_vars = env_arr.shape
    
    # Center X
    X_c = X_arr - np.mean(X_arr, axis=0)
    
    # Pre-calculate inverse of X'X for regression beta
    # To handle potential collinearity or rank deficiency, use pseudo-inverse
    XtX_inv_Xt = np.linalg.pinv(X_c)
    
    vectors = np.zeros((n_vars, n_dims))
    r2_vals = np.zeros(n_vars)
    p_vals = np.zeros(n_vars)
    
    for v in range(n_vars):
        y = env_arr[:, v]
        y_c = y - np.mean(y)
        ss_y = np.sum(y_c**2)
        
        if ss_y == 0:
            p_vals[v] = np.nan
            continue
            
        beta = XtX_inv_Xt @ y_c
        
        # normalize beta for direction cosines
        norm_beta = np.sqrt(np.sum(beta**2))
        if norm_beta > 0:
            vectors[v, :] = beta / norm_beta
        
        y_fit = X_c @ beta
        ss_res = np.sum((y_c - y_fit)**2)
        r2_obs = 1.0 - (ss_res / ss_y)
        r2_vals[v] = r2_obs
        
        greater_eq = 1
        for _ in range(permutations):
            y_perm = np.random.permutation(y_c)
            beta_perm = XtX_inv_Xt @ y_perm
            y_fit_perm = X_c @ beta_perm
            ss_res_perm = np.sum((y_perm - y_fit_perm)**2)
            r2_perm = 1.0 - (ss_res_perm / ss_y)
            
            if r2_perm >= r2_obs - 1e-12:
                greater_eq += 1
                
        p_vals[v] = greater_eq / (permutations + 1)
        
    return {
        "vectors": vectors,
        "r2": r2_vals,
        "pvals": p_vals
    }

def cca(x: Union[np.ndarray, pd.DataFrame]) -> dict:
    """
    Correspondence Analysis (Unconstrained CCA).
    
    Mimics `vegan::cca(x)`. Computes the chi-squared distances and extracts 
    the principal inertias (eigenvalues) of the community matrix.
    
    Parameters
    ----------
    x : array-like
        Community data matrix. Must contain non-negative values.
        
    Returns
    -------
    dict
        "eigenvalues": Array of principal inertias.
        "tot.chi": Total inertia (variance).
    """
    x_arr = np.asarray(x, dtype=float)
    if np.any(x_arr < 0):
        raise ValueError("Correspondence analysis requires non-negative data.")
        
    tot = np.sum(x_arr)
    if tot == 0:
        raise ValueError("Sum of data matrix must be > 0.")
        
    # Relative frequencies
    P = x_arr / tot
    
    # Margins
    r = np.sum(P, axis=1)
    c = np.sum(P, axis=0)
    
    # Remove empty rows/columns
    r_mask = r > 0
    c_mask = c > 0
    
    P_sub = P[r_mask][:, c_mask]
    r_sub = r[r_mask]
    c_sub = c[c_mask]
    
    # Expected
    expected = np.outer(r_sub, c_sub)
    
    # Standardized residuals
    Z = (P_sub - expected) / np.sqrt(expected)
    
    # SVD
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    
    eigenvalues = S ** 2
    # filter tiny values
    eigenvalues = eigenvalues[eigenvalues > 1e-12]
    
    # Total inertia: sum of squared elements of Z
    tot_chi = np.sum(Z ** 2)
    
    return {
        "eigenvalues": eigenvalues,
        "tot.chi": tot_chi
    }

def wascores(x: Union[np.ndarray, pd.DataFrame], 
             w: Union[np.ndarray, pd.DataFrame], 
             expand: bool = False) -> pd.DataFrame:
    """
    Weighted Averages Scores for Species.
    
    Mimics `vegan::wascores`. Computes the weighted averages of site scores 
    for species.
    
    Parameters
    ----------
    x : array-like
        Ordination site scores (n_sites, n_dimensions).
    w : array-like
        Community abundances matrix (n_sites, n_species).
    expand : bool
        If True, expand scores so that their variance equals the variance of the 
        site scores (equivalent to vegan's `expand=TRUE`).
        
    Returns
    -------
    pd.DataFrame
        Species scores (n_species, n_dimensions).
    """
    x_arr = np.asarray(x, dtype=float)
    w_arr = np.asarray(w, dtype=float)
    
    # Check dimensions
    if x_arr.shape[0] != w_arr.shape[0]:
        raise ValueError("Number of rows in site scores must match number of rows in community abundances.")
        
    # w.T (species x sites) @ x (sites x dims) -> (species x dims)
    # divided by sum of abundances per species
    col_sums = np.sum(w_arr, axis=0)[:, np.newaxis]
    
    with np.errstate(divide='ignore', invalid='ignore'):
        wa = np.where(col_sums > 0, w_arr.T.dot(x_arr) / col_sums, 0)
        
    if expand:
        # Variance of sites
        var_sites = np.var(x_arr, axis=0, ddof=1)
        # Variance of WA scores
        var_wa = np.var(wa, axis=0, ddof=1)
        
        # Scaling factor: sqrt(var_sites / var_wa)
        with np.errstate(divide='ignore', invalid='ignore'):
            scale = np.where(var_wa > 0, np.sqrt(var_sites / var_wa), 1.0)
            
        wa = wa * scale
        
    species_names = [f"Sp{i+1}" for i in range(w_arr.shape[1])]
    if isinstance(w, pd.DataFrame):
        species_names = w.columns
        
    dim_names = [f"Dim{i+1}" for i in range(x_arr.shape[1])]
    if isinstance(x, pd.DataFrame):
        dim_names = x.columns
        
    return pd.DataFrame(wa, index=species_names, columns=dim_names)

def isomap(dist: Union[np.ndarray, pd.DataFrame], 
           k: int = 5, 
           ndim: int = 2) -> dict:
    """
    Isomap Ordination.
    
    Mimics `vegan::isomap`. Constructs a nearest-neighbors geodesic network 
    and returns its lower-dimensional scaling coordinates.
    
    Parameters
    ----------
    dist : array-like
        Precomputed distance matrix (condensed or square 2D).
    k : int
        Number of nearest neighbors to construct the geodesic network.
    ndim : int
        Number of dimensions for the configuration.
        
    Returns
    -------
    dict
        "points": Ordination coordinates (n_sites, ndim).
    """
    from sklearn.manifold import Isomap as SklearnIsomap
    dist_arr = np.asarray(dist, dtype=float)
    
    if dist_arr.ndim == 1:
        dist_sq = squareform(dist_arr)
    else:
        dist_sq = dist_arr
        
    iso = SklearnIsomap(n_neighbors=k, n_components=ndim, metric="precomputed")
    points = iso.fit_transform(dist_sq)
    
    # In vegan, it also returns the net/geodesic distance matrix.
    # scikit-learn stores the geodesic distance graph in `dist_matrix_` on the fitted object.
    
    return {
        "points": points,
        "dist": iso.dist_matrix_
    }

def capscale(dist: Union[np.ndarray, pd.DataFrame], 
             Y: Union[np.ndarray, pd.DataFrame]) -> dict:
    """
    Constrained Analysis of Principal Coordinates (Distance-based RDA).
    
    Mimics `vegan::capscale` allowing Euclidean constrained regression 
    over non-Euclidean functional coordinate projections.
    
    Parameters
    ----------
    dist : array-like
        Precomputed distance matrix.
    Y : array-like
        Predictor matrix (constraints).
        
    Returns
    -------
    dict
        Same format as `rda` ("CCA", "CA", "tot.chi").
    """
    dist_arr = np.asarray(dist, dtype=float)
    
    # Squareform ensures we can extract dimensions 
    if dist_arr.ndim == 1:
        dist_sq = squareform(dist_arr)
    else:
        dist_sq = dist_arr
        
    n_sites = dist_sq.shape[0]
    
    # Compute PCoA points extracting fully positive Euclidean mappings
    points, eig = cmdscale(dist_sq, k=n_sites - 1)
    
    # Pass derived Euclidean spatial configurations into explicit RDA
    return rda(points, Y)

def anova_rda(x: Union[np.ndarray, pd.DataFrame], 
              Y: Union[np.ndarray, pd.DataFrame], 
              permutations: int = 999) -> dict:
    """
    Permutation Analysis of Variance for Constrained Ordinations.
    
    Mimics `vegan::anova.cca` behavior specifically for Redundancy Analysis.
    Evaluates the pseudo-F statistic generated by constrained inertia vs 
    unconstrained residual inertia, mapping significance across permutations.
    
    Parameters
    ----------
    x : array-like
        Community data matrix.
    Y : array-like
        Predictor matrix (constraints).
    permutations : int
        Number of matrix row-permutations computing pseudo-F distributions.
        
    Returns
    -------
    dict
        "F": Observed pseudo-F statistic.
        "p_value": Empirical permutation p-value mapping statistical significance.
        "df_model": Degrees of Freedom for the constraints.
        "df_res": Degrees of freedom for configurations strictly orthogonal to Y.
    """
    obs_res = rda(x, Y)
    
    var_cca = obs_res["CCA"]["tot.chi"]
    var_ca = obs_res["CA"]["tot.chi"]
    
    n_sites = np.asarray(x).shape[0]
    q_vars = np.asarray(Y).shape[1]
    
    df_cca = q_vars
    df_ca = n_sites - q_vars - 1
    
    if df_cca == 0 or df_ca == 0:
        return {"F": np.nan, "p_value": np.nan, "df_model": df_cca, "df_res": df_ca}
        
    f_obs = (var_cca / df_cca) / (var_ca / df_ca)
    
    y_arr = np.asarray(Y)
    perm_f = np.zeros(permutations)
    
    for i in range(permutations):
        # Independently restructure constraints against native community distributions
        y_perm = np.random.permutation(y_arr)
        perm_res = rda(x, y_perm)
        
        p_var_cca = perm_res["CCA"]["tot.chi"]
        p_var_ca = perm_res["CA"]["tot.chi"]
        
        perm_f[i] = (p_var_cca / df_cca) / (p_var_ca / df_ca)
        
    p_val = (np.sum(perm_f >= f_obs) + 1.0) / (permutations + 1.0)
    
    return {
        "F": f_obs,
        "p_value": p_val,
        "df_model": df_cca,
        "df_res": df_ca
    }

def pca(x: Union[np.ndarray, pd.DataFrame]) -> dict:
    """
    Principal Component Analysis wrapper.
    
    Syntactic wrapper identical structurally mapping boundaries resolving to 
    `vegan::rda(x)` mathematically without hierarchical constraints.
    """
    return rda(x)

def tolerance(x: np.ndarray, site_scores: np.ndarray, species_scores: np.ndarray) -> np.ndarray:
    """
    Species Niche Tolerance Mapping.
    
    Mimics `vegan::tolerance`. Calculates the root mean squared deviation from 
    species optima (centroids) structurally extracting environmental niche widths natively!
    """
    x_arr = np.asarray(x, dtype=float)
    x_sum = np.sum(x_arr, axis=0)
    
    out = np.zeros_like(species_scores)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        for k in range(site_scores.shape[1]):
            diff = site_scores[:, k].reshape(-1, 1) - species_scores[:, k].reshape(1, -1)
            weighted_sq_diff = x_arr * (diff ** 2)
            tol_var = np.sum(weighted_sq_diff, axis=0) / x_sum 
            out[:, k] = np.sqrt(tol_var)
            
    return np.nan_to_num(out, nan=0.0)

def varpart(x: Union[np.ndarray, pd.DataFrame], *env_matrices: Union[np.ndarray, pd.DataFrame]) -> dict:
    """
    Variation Partitioning evaluating Adjusted R-square limits.
    
    Mimics `vegan::varpart`. Evaluates explicit Unique and Shared constraints inherently natively 
    by structuring continuous recursive redundancy iterations explicitly.
    Supports exactly 2 explicit structural variable components natively.
    """
    x_arr = np.asarray(x, dtype=float)
    n = x_arr.shape[0]
    
    num_mats = len(env_matrices)
    if num_mats != 2:
        raise ValueError("varpart currently natively supports exactly 2 explicitly matrix structures for $R^2$ scaling.")
        
    def _adj_r2(Y):
        y_arr = np.asarray(Y, dtype=float)
        try:
            res = rda(x_arr, y_arr)
            r2 = res["CCA"]["tot.chi"] / res["tot.chi"]
            p = np.linalg.matrix_rank(y_arr)
            if n - p - 1 <= 0:
                return 0.0
            return 1.0 - (1.0 - r2) * ((n - 1) / (n - p - 1))
        except:
            return 0.0
            
    y1, y2 = [np.asarray(y, dtype=float) for y in env_matrices]
    y12 = np.hstack([y1, y2])
    
    r1 = _adj_r2(y1)
    r2 = _adj_r2(y2)
    r12 = _adj_r2(y12)
    
    a = r12 - r2
    c = r12 - r1
    b = r1 - a
    d = 1.0 - r12
    
    df_out = pd.DataFrame({
        "R2.adj": [r1, r2, r12, a, b, c, d]
    }, index=["X1", "X2", "X1+X2", "Unique_X1", "Shared", "Unique_X2", "Unexplained"])
    
    return {"fractions": df_out}

def prc(response: Union[np.ndarray, pd.DataFrame], 
        treatment: Union[np.ndarray, list, pd.Series], 
        time: Union[np.ndarray, list, pd.Series]) -> dict:
    """
    Principal Response Curves.
    
    Mimics `vegan::prc` parameterizing constraints uniquely isolating Time*Treatment 
    interactions structurally explicitly mapping partial RDA.
    
    Parameters
    ----------
    response : array-like
        Community data matrix.
    treatment : array-like
        Categorical array identifying independent treatment blocks.
    time : array-like
        Categorical array identifying longitudinal timeline mappings.
        
    Returns
    -------
    dict
        Equivalent dictionary to partial RDA mapping variables evaluating correctly.
    """
    time_dummies = pd.get_dummies(time, drop_first=True, dtype=float)
    trt_dummies = pd.get_dummies(treatment, drop_first=True, dtype=float)
    
    Z = time_dummies.values
    n = np.asarray(response).shape[0]
    interactions = []
    
    for t_col in time_dummies.columns:
        for tr_col in trt_dummies.columns:
            inter = time_dummies[t_col].values * trt_dummies[tr_col].values
            interactions.append(inter)
            
    if not interactions:
        Y = np.zeros((n, 1))
    else:
        Y = np.column_stack(interactions)
        
    X_arr = np.asarray(response, dtype=float)
    X_cent = X_arr - np.mean(X_arr, axis=0)
    
    # Add intercept to condition matrix Z
    Z_full = np.column_stack([np.ones(n), Z])
    try:
        Q, _ = np.linalg.qr(Z_full)
        X_res = X_cent - Q @ (Q.T @ X_cent)
        # Partial Y too natively
        Y_res = Y - Q @ (Q.T @ Y)
    except:
        X_res = X_cent
        Y_res = Y
        
    return rda(X_res, Y_res)



