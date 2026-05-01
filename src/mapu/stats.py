import numpy as np
import pandas as pd
from typing import Union
from scipy.stats import rankdata
from mapu.vegdist import vegdist

def anosim(x: Union[np.ndarray, pd.DataFrame], 
           grouping: np.ndarray, 
           distance: str = "bray", 
           permutations: int = 999) -> dict:
    """
    Analysis of similarities (ANOSIM).
    
    Mimics `vegan::anosim`.
    
    Parameters
    ----------
    x : array-like
        Community data matrix.
    grouping : array-like
        Factor for grouping observations.
    distance : str, default "bray"
        Distance metric to use.
    permutations : int, default 999
        Number of permutations to assess significance.
        
    Returns
    -------
    dict
        A dictionary containing:
        - "statistic": The R statistic
        - "significance": The p-value
        - "permutations": Total number of permutations
    """
    # 1. Compute distance matrix
    dist_vector = vegdist(x, method=distance, upper=False)
    
    # 2. Get the condensed index mappings for pair (i, j)
    # The condensed distance matrix from `pdist` or `vegdist` length is N*(N-1)/2
    n = x.shape[0] if isinstance(x, np.ndarray) else len(x)
    
    if len(grouping) != n:
        raise ValueError("Length of grouping must match number of observations in x.")
        
    grouping = np.asarray(grouping)
    
    # Create masks for within-group distances
    # A distance is within-group if grouping[i] == grouping[j]
    ii, jj = np.triu_indices(n, k=1)
    within_mask = (grouping[ii] == grouping[jj])
    between_mask = ~within_mask
    
    # 3. Rank distances
    ranks = rankdata(dist_vector)
    
    def calc_R(ranks, within, between, n):
        r_w = np.mean(ranks[within]) if np.any(within) else 0.0
        r_b = np.mean(ranks[between]) if np.any(between) else 0.0
        # R = (r_b - r_w) / (N * (N - 1) / 4)
        denominator = (n * (n - 1)) / 4.0
        return (r_b - r_w) / denominator
        
    R_obs = calc_R(ranks, within_mask, between_mask, n)
    
    # 4. Permutations
    greater_eq_count = 1  # Includes the observed value
    
    for _ in range(permutations):
        grouping_perm = grouping.copy()
        np.random.shuffle(grouping_perm)
        # Re-compute within mask
        p_within_mask = (grouping_perm[ii] == grouping_perm[jj])
        p_between_mask = ~p_within_mask
        
        R_perm = calc_R(ranks, p_within_mask, p_between_mask, n)
        if R_perm >= R_obs:
            greater_eq_count += 1
            
    p_value = greater_eq_count / (permutations + 1)
    
    return {
        "statistic": R_obs,
        "significance": p_value,
        "permutations": permutations
    }

from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import squareform

def mantel(xdis: np.ndarray, 
           ydis: np.ndarray, 
           method: str = "pearson", 
           permutations: int = 999) -> dict:
    """
    Mantel statistic to test correlation between two distance matrices.
    
    Mimics `vegan::mantel`.
    
    Parameters
    ----------
    xdis : array-like
        Distance matrix 1 (condensed or square).
    ydis : array-like
        Distance matrix 2 (condensed or square).
    method : str
        "pearson" or "spearman".
    permutations : int
        Number of permutations to assess significance.
        
    Returns
    -------
    dict
        "statistic": Mantel r statistic
        "significance": p-value
        "permutations": number of permutations
    """
    xdis = np.asarray(xdis)
    ydis = np.asarray(ydis)
    
    if xdis.ndim == 2:
        x_cond = squareform(xdis)
        x_sq = xdis
    else:
        x_cond = xdis
        x_sq = squareform(xdis)
        
    if ydis.ndim == 2:
        y_cond = squareform(ydis)
        y_sq = ydis
    else:
        y_cond = ydis
        y_sq = squareform(ydis)
        
    if len(x_cond) != len(y_cond):
        raise ValueError("Distance matrices must have the same number of elements.")
        
    n = x_sq.shape[0]
    
    def get_corr(x, y):
        if method == "pearson":
            return pearsonr(x, y)[0]
        elif method == "spearman":
            return spearmanr(x, y)[0]
        else:
            raise ValueError("Method must be 'pearson' or 'spearman'")
            
    r_obs = get_corr(x_cond, y_cond)
    
    greater_eq_count = 1
    
    for _ in range(permutations):
        perm = np.random.permutation(n)
        # Permute rows and columns of y_sq
        y_sq_perm = y_sq[np.ix_(perm, perm)]
        y_cond_perm = squareform(y_sq_perm)
        
        r_perm = get_corr(x_cond, y_cond_perm)
        if r_perm >= r_obs:
            greater_eq_count += 1
            
    p_value = greater_eq_count / (permutations + 1)
    
    return {
        "statistic": r_obs,
        "significance": p_value,
        "permutations": permutations
    }

def adonis(x: Union[np.ndarray, pd.DataFrame], 
           grouping: np.ndarray, 
           distance: str = "bray", 
           permutations: int = 999) -> dict:
    """
    Permutational Multivariate Analysis of Variance (PERMANOVA).
    
    Mimics a one-way `vegan::adonis` or `vegan::adonis2`.
    
    Parameters
    ----------
    x : array-like
        Community data matrix.
    grouping : array-like
        Factor for grouping observations.
    distance : str, default "bray"
        Distance metric to use.
    permutations : int, default 999
        Number of permutations to assess significance.
        
    Returns
    -------
    dict
        "F.Model": Pseudo-F statistic
        "R2": R-squared value
        "significance": p-value
        "permutations": total permutations used
    """
    dist_vector = vegdist(x, method=distance, upper=False)
    
    n = x.shape[0] if isinstance(x, np.ndarray) else len(x)
    if len(grouping) != n:
        raise ValueError("Length of grouping must match number of observations.")
    
    grouping = np.asarray(grouping)
    groups = np.unique(grouping)
    g = len(groups)
    
    if g == 1:
        raise ValueError("Grouping factor must have at least 2 levels.")
        
    # distances squared
    d2 = dist_vector ** 2
    sst = np.sum(d2) / n
    
    def calc_ssw(grps):
        ssw = 0.0
        ii, jj = np.triu_indices(n, k=1)
        for val in groups:
            # Mask for elements where both row and column belong to this group
            mask = (grps[ii] == val) & (grps[jj] == val)
            n_k = np.sum(grps == val)
            if n_k > 0:
                ssw += np.sum(d2[mask]) / n_k
        return ssw
        
    ssw_obs = calc_ssw(grouping)
    ssb_obs = sst - ssw_obs
    
    df_between = g - 1
    df_within = n - g
    
    if ssw_obs == 0:
        f_obs = np.inf
    else:
        f_obs = (ssb_obs / df_between) / (ssw_obs / df_within)
        
    r2_obs = ssb_obs / sst
    
    greater_eq_count = 1
    
    for _ in range(permutations):
        grps_perm = grouping.copy()
        np.random.shuffle(grps_perm)
        ssw_perm = calc_ssw(grps_perm)
        ssb_perm = sst - ssw_perm
        f_perm = (ssb_perm / df_between) / (ssw_perm / df_within) if ssw_perm > 0 else np.inf
        
        if f_perm >= f_obs:
            greater_eq_count += 1
            
    p_value = greater_eq_count / (permutations + 1)
    
    return {
        "F.Model": f_obs,
        "R2": r2_obs,
        "significance": p_value,
        "permutations": permutations
    }

def mrpp(x: Union[np.ndarray, pd.DataFrame], 
         grouping: np.ndarray, 
         distance: str = "bray", 
         permutations: int = 999) -> dict:
    """
    Multi-Response Permutation Procedures (MRPP).
    
    Mimics `vegan::mrpp`.
    
    Parameters
    ----------
    x : array-like
        Community data matrix.
    grouping : array-like
        Factor for grouping observations.
    distance : str, default "bray"
        Distance metric to use.
    permutations : int, default 999
        Number of permutations to assess significance.
        
    Returns
    -------
    dict
        "delta": Observed delta (weighted mean within-group distance)
        "E.delta": Expected delta
        "significance": p-value
    """
    dist_vector = vegdist(x, method=distance, upper=False)
    n = x.shape[0] if isinstance(x, np.ndarray) else len(x)
    grouping = np.asarray(grouping)
    groups = np.unique(grouping)
    
    # Pre-calculate pairwise weights
    ii, jj = np.triu_indices(n, k=1)
    
    def calc_delta(grps):
        delta = 0.0
        for val in groups:
            mask = (grps[ii] == val) & (grps[jj] == val)
            n_k = np.sum(grps == val)
            if n_k > 1:
                # weight is n_k / n (default vegan weight.type=1)
                weight = n_k / n
                mean_dist = np.mean(dist_vector[mask]) if np.any(mask) else 0.0
                delta += weight * mean_dist
        return delta
        
    delta_obs = calc_delta(grouping)
    
    greater_eq_count = 1
    delta_perms = []
    
    for _ in range(permutations):
        grps_perm = grouping.copy()
        np.random.shuffle(grps_perm)
        delta_perm = calc_delta(grps_perm)
        delta_perms.append(delta_perm)
        if delta_perm <= delta_obs:  # MRPP tests if observed is smaller than expected
            greater_eq_count += 1
            
    p_value = greater_eq_count / (permutations + 1)
    e_delta = np.mean(delta_perms)
    
    return {
        "delta": delta_obs,
        "E.delta": e_delta,
        "significance": p_value
    }


def simper(x: pd.DataFrame, grouping: np.ndarray) -> dict:
    """
    Similarity Percentages.
    
    Mimics `vegan::simper` using Bray-Curtis.
    Finds the average contribution of each species to the dissimilarity 
    between two groups. For simplicity, tests between the first two unique groups found.
    
    Parameters
    ----------
    x : pd.DataFrame
        Community data matrix.
    grouping : array-like
        Factor for grouping.
        
    Returns
    -------
    dict
        Dictionary containing average species contributions.
    """
    x_arr = np.asarray(x, dtype=float)
    grouping = np.asarray(grouping)
    groups = np.unique(grouping)
    
    if len(groups) < 2:
        raise ValueError("simper requires at least 2 groups")
        
    # Pick first two groups
    g1 = groups[0]
    g2 = groups[1]
    
    x1 = x_arr[grouping == g1]
    x2 = x_arr[grouping == g2]
    
    n_species = x_arr.shape[1]
    contributions = np.zeros(n_species)
    count = 0
    
    # Calculate for each pair between g1 and g2
    for i in range(x1.shape[0]):
        for j in range(x2.shape[0]):
            sum_total = np.sum(x1[i] + x2[j])
            if sum_total > 0:
                diffs = np.abs(x1[i] - x2[j])
                contributions += diffs / sum_total
            count += 1
            
    if count > 0:
        contributions /= count
        
    return {col: val for col, val in zip(x.columns, contributions)}

def betadisper(x: pd.DataFrame, 
               grouping: np.ndarray, 
               distance: str = "bray") -> dict:
    """
    Multivariate homogeneity of groups dispersions.
    
    Mimics `vegan::betadisper`. Evaluates the distances to group centroids.
    Uses Anderson (2006) direct calculation without full SVD coordinates.
    
    Parameters
    ----------
    x : pd.DataFrame
        Community data matrix.
    grouping : array-like
        Factor for grouping.
    distance : str, default "bray"
        Distance metric to use.
        
    Returns
    -------
    dict
        "distances": array of distances to centroid for each site.
    """
    dist_vector = vegdist(x, method=distance, upper=False)
    n = x.shape[0] if isinstance(x, np.ndarray) else len(x)
    
    # Square the condensed distances and build a square matrix for easy indexing
    d_sq_cond = dist_vector ** 2
    d_sq = squareform(d_sq_cond)
    
    grouping = np.asarray(grouping)
    groups = np.unique(grouping)
    
    z = np.zeros(n)
    
    for val in groups:
        # idx is boolean array for elements in group 'val'
        idx = (grouping == val)
        n_k = np.sum(idx)
        
        if n_k > 0:
            # Extract subgroup squared distance matrix
            subset_d2 = d_sq[np.ix_(idx, idx)]
            
            # Sum of upper triangle of subgroup, which is sum_{j<m} d_{jm}^2
            # because subset_d2 is symmetric with 0 diagonal, we can just sum all and divide by 2
            sum_all_pairs = np.sum(subset_d2) / 2.0
            
            # The second term: 1/n^2 sum d_{jm}^2
            correction = sum_all_pairs / (n_k ** 2)
            
            # First term: 1/n sum_j d_{ij}^2
            # For each i in the group, we sum across its row
            row_sums = np.sum(subset_d2, axis=1) / n_k
            
            # z_i^2
            z_sq = row_sums - correction
            
            # Handle potential negative float precision errors near 0
            z_sq = np.maximum(z_sq, 0)
            
            z[idx] = np.sqrt(z_sq)
            
    return {
        "distances": z
    }

def meandist(dist: Union[np.ndarray, pd.DataFrame], grouping: np.ndarray) -> pd.DataFrame:
    """
    Mean distance within and between groups.
    
    Mimics `vegan::meandist`.
    Finds the mean within-group distances (diagonal) and between-group distances (off-diagonal).
    
    Parameters
    ----------
    dist : array-like
        A square distance matrix or condensed distance array.
    grouping : array-like
        Factor for grouping.
        
    Returns
    -------
    pd.DataFrame
        A symmetric GxG matrix containing the mean distances.
    """
    from scipy.spatial.distance import is_valid_y, squareform
    
    dist_arr = np.asarray(dist, dtype=float)
    if dist_arr.ndim == 1:
        if not is_valid_y(dist_arr):
            raise ValueError("Input 1D array is not a valid condensed distance matrix.")
        d_sq = squareform(dist_arr)
    else:
        d_sq = dist_arr
        
    grouping = np.asarray(grouping)
    groups = np.unique(grouping)
    g = len(groups)
    
    out = np.zeros((g, g))
    
    for i in range(g):
        for j in range(i, g):
            idx_i = (grouping == groups[i])
            idx_j = (grouping == groups[j])
            
            submat = d_sq[np.ix_(idx_i, idx_j)]
            
            if i == j:
                # Within group: mean of upper triangle (excluding diagonal 0s)
                n_k = np.sum(idx_i)
                if n_k > 1:
                    # extract upper triangle elements
                    triu_elements = submat[np.triu_indices(n_k, k=1)]
                    val = np.mean(triu_elements)
                else:
                    val = 0.0
            else:
                # Between group: mean of entire submatrix block
                val = np.mean(submat)
                
            out[i, j] = val
            out[j, i] = val
            
    return pd.DataFrame(out, index=groups, columns=groups)

from itertools import combinations

def bioenv(x: Union[np.ndarray, pd.DataFrame], 
           env: pd.DataFrame, 
           method: str = "spearman", 
           index: str = "bray",
           max_vars: int = None) -> dict:
    """
    Best subset of environmental variables with maximum correlation 
    with community dissimilarities.
    
    Mimics `vegan::bioenv`.
    
    Parameters
    ----------
    x : array-like
        Community data matrix.
    env : pd.DataFrame
        Continuous environmental variables.
    method : str
        Correlation method: "spearman" (default) or "pearson".
    index : str
        Dissimilarity index for community data.
    max_vars : int
        Maximum number of variables in a subset. Default is all.
        
    Returns
    -------
    dict
        "max_corr": Maximum correlation achieved.
        "best_subset": Tuple of the best environmental variable indices (0-indexed).
        "best_subset_names": List of names if env is a DataFrame.
    """
    x_arr = np.asarray(x, dtype=float)
    is_df = isinstance(env, pd.DataFrame)
    env_arr = np.asarray(env, dtype=float)
    
    n_sites, n_env = env_arr.shape
    
    if max_vars is None:
        max_vars = n_env
    
    # Pre-scale environmental data to unit variance (classic vegan behavior)
    env_scaled = (env_arr - np.mean(env_arr, axis=0)) / np.std(env_arr, axis=0, ddof=1)
    
    # 1. Base Community Distance (condensed)
    dx = vegdist(x_arr, method=index, upper=False)
    
    def get_corr(d1, d2):
        if method == "pearson":
            return pearsonr(d1, d2)[0]
        else:
            return spearmanr(d1, d2)[0]
    
    best_corr = -1.0
    best_combo = ()
    
    from scipy.spatial.distance import pdist
    
    for k in range(1, max_vars + 1):
        for combo in combinations(range(n_env), k):
            sub_env = env_scaled[:, combo]
            dy = pdist(sub_env, metric="euclidean")
            
            corr = get_corr(dx, dy)
            if corr > best_corr:
                best_corr = corr
                best_combo = combo
                
    result = {
        "max_corr": best_corr,
        "best_subset": best_combo,
    }
    
    if is_df:
        result["best_subset_names"] = [env.columns[i] for i in best_combo]
        
    return result

def permatfull(m: Union[np.ndarray, pd.DataFrame], 
               times: int = 1) -> list:
    """
    Null models for community matrices (unrestricted).
    
    Mimics `vegan::permatfull(..., fixedmar="none", mtype="count")`.
    Generates completely random permutations of the given matrix.
    
    Parameters
    ----------
    m : array-like
        Community data matrix.
    times : int
        Number of permuted matrices to generate.
        
    Returns
    -------
    list
        List of randomly permuted arrays/DataFrames.
    """
    m_arr = np.asarray(m)
    shape = m_arr.shape
    is_df = isinstance(m, pd.DataFrame)
    
    perms = []
    
    # Unrestricted permutation
    for _ in range(times):
        flat = m_arr.flatten()
        np.random.shuffle(flat)
        perm = flat.reshape(shape)
        
        if is_df:
            perm = pd.DataFrame(perm, index=m.index, columns=m.columns)
            
        perms.append(perm)
        
    return perms

def oecosimu(x: Union[np.ndarray, pd.DataFrame], 
             statistic: callable, 
             method: str = "permatfull", 
             nsimul: int = 99) -> dict:
    """
    Evaluate a statistic against a null community model.
    
    Mimics `vegan::oecosimu`. Generates `nsimul` permutations of `x` using 
    the specified null model method, evaluates `statistic()` on both the 
    observed matrix and permutations, and computes standard Z-scores and means.
    
    Parameters
    ----------
    x : array-like
        Community data matrix.
    statistic : callable
        A function that takes a community matrix and returns a numeric array or scalar.
    method : str
        Null model method. Only "permatfull" is supported currently.
    nsimul : int
        Number of permutations.
        
    Returns
    -------
    dict
        "statistic": Observed statistic values.
        "means": Mean of simulated statistics.
        "z": Z-scores of the observed vs simulated distributions.
    """
    if method != "permatfull":
        raise ValueError("Only 'permatfull' null models are currently supported inside oecosimu.")
        
    obs_stat = np.asarray(statistic(x), dtype=float)
    
    perms = permatfull(x, times=nsimul)
    sim_stats = []
    
    for perm in perms:
        sim_stats.append(np.asarray(statistic(perm), dtype=float))
        
    sim_stack = np.stack(sim_stats, axis=0) # (nsimul, ... shape of stat ...)
    
    sim_means = np.nanmean(sim_stack, axis=0)
    sim_stds = np.nanstd(sim_stack, axis=0, ddof=1)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        z_scores = np.where(sim_stds > 0, (obs_stat - sim_means) / sim_stds, 0.0)
        
    return {
        "statistic": obs_stat,
        "means": sim_means,
        "z": z_scores
    }

def indval(x: Union[np.ndarray, pd.DataFrame], 
           group: Union[np.ndarray, list, pd.Series]) -> dict:
    """
    Indicator Species Analysis (Dufrene & Legendre 1997).
    
    Mimics `labdsv::indval` or `multipatt` assigning indicator limits 
    mapping species representations across cluster groupings.
    
    Parameters
    ----------
    x : array-like
        Community data matrix.
    group : array-like
        Group assignments per site (categorical vector).
        
    Returns
    -------
    dict
        "indval": Indicator values for each Species x Group mapping.
        "A": Specificity allocations. 
        "B": Fidelity distributions.
    """
    x_arr = np.asarray(x, dtype=float)
    g_arr = np.asarray(group)
    
    if x_arr.shape[0] != len(g_arr):
        raise ValueError("Group array length must match number of sites.")
        
    unique_groups = np.unique(g_arr)
    n_groups = len(unique_groups)
    n_species = x_arr.shape[1]
    
    A = np.zeros((n_groups, n_species))
    B = np.zeros((n_groups, n_species))
    
    # Pre-compute totals for A
    group_means = np.zeros((n_groups, n_species))
    for i, g in enumerate(unique_groups):
        idx = (g_arr == g)
        group_means[i, :] = np.mean(x_arr[idx, :], axis=0)
        
    sum_means = np.sum(group_means, axis=0)
    
    for i, g in enumerate(unique_groups):
        idx = (g_arr == g)
        n_g = np.sum(idx)
        
        # A: Specificity (mean abundance in group / sum of mean abundances across groups)
        with np.errstate(divide='ignore', invalid='ignore'):
            A[i, :] = np.where(sum_means > 0, group_means[i, :] / sum_means, 0)
            
        # B: Fidelity (proportion of sites in group where species is present)
        b_present = (x_arr[idx, :] > 0).astype(float)
        B[i, :] = np.sum(b_present, axis=0) / n_g if n_g > 0 else 0
        
    ind_vals = A * B * 100.0
    
    col_names = x.columns if isinstance(x, pd.DataFrame) else [f"Sp_{i}" for i in range(n_species)]
    group_names = [str(g) for g in unique_groups]
    
    return {
        "indval": pd.DataFrame(ind_vals, index=group_names, columns=col_names),
        "A": pd.DataFrame(A, index=group_names, columns=col_names),
        "B": pd.DataFrame(B, index=group_names, columns=col_names)
    }

def permatswap(x: Union[np.ndarray, pd.DataFrame], 
               times: int = 99,
               burnin: int = 0) -> list:
    """
    Sequential Swap for Occurrence Matrices.
    
    Mimics `vegan::permatswap` algorithm conserving perfectly both 
    species prevalence (column sums) and plot richness (row sums).
    Returns list of permuted matrices.
    
    Parameters
    ----------
    x : array-like
        Community data matrix.
    times : int
        Number of valid permuted arrays to return sequentially.
    burnin : int
        Optional burn sequences dropped to ensure randomization mixing.
        
    Returns
    -------
    list
        List of generated 2D matrices.
    """
    x_arr = np.asarray(x, dtype=float)
    b_arr = (x_arr > 0).astype(int)
    
    rows, cols = np.where(b_arr == 1)
    n_ones = len(rows)
    
    out_matrices = []
    current_mat = b_arr.copy()
    
    import random
    
    for _ in range(burnin + times):
        # Conduct robust sequential sub-swaps ensuring matrix mixing
        n_swaps = n_ones * 10
        for _ in range(n_swaps):
            if n_ones < 2:
                break
                
            idx1 = random.randrange(n_ones)
            idx2 = random.randrange(n_ones)
            while idx1 == idx2:
                idx2 = random.randrange(n_ones)
                
            r1, c1 = rows[idx1], cols[idx1]
            r2, c2 = rows[idx2], cols[idx2]
            
            if r1 != r2 and c1 != c2:
                if current_mat[r1, c2] == 0 and current_mat[r2, c1] == 0:
                    current_mat[r1, c1] = 0
                    current_mat[r2, c2] = 0
                    current_mat[r1, c2] = 1
                    current_mat[r2, c1] = 1
                    
                    cols[idx1] = c2
                    cols[idx2] = c1
                    
        out_matrices.append(current_mat.copy())
        
    return out_matrices[burnin:]

def mantel_correlog(D_eco: Union[np.ndarray, pd.DataFrame], 
                    D_geo: Union[np.ndarray, pd.DataFrame],
                    n_classes: int = 10,
                    permutations: int = 99) -> pd.DataFrame:
    """
    Mantel Correlogram.
    
    Mimics `vegan::mantel.correlog`. Sequentially structures nested bins natively 
    iterating Mantel significance mappings establishing spatial correlation decay cleanly.
    
    Parameters
    ----------
    D_eco : array-like
        Ecological distance distributions.
    D_geo : array-like
        Geographical distances.
    n_classes : int
        Number of correlation bins.
    permutations : int
        Number of significance testing permutations linearly.
        
    Returns
    -------
    pd.DataFrame
        Bin evaluations mapped strictly generating decay profiles sequentially.
    """
    D_e = np.asarray(D_eco, dtype=float)
    D_g = np.asarray(D_geo, dtype=float)
    
    if D_e.ndim == 2 and D_e.shape[0] == D_e.shape[1]:
        D_e = np.asarray(squareform(D_e, checks=False))
    if D_g.ndim == 2 and D_g.shape[0] == D_g.shape[1]:
        D_g = np.asarray(squareform(D_g, checks=False))
        
    bins = np.linspace(np.min(D_g), np.max(D_g), n_classes + 1)
    results = []
    
    for i in range(n_classes):
        lower = bins[i]
        upper = bins[i+1]
        
        # 0 if in bin, 1 outside (creates standard structure correlating similarities linearly)
        mask_bins = (~((D_g >= lower) & (D_g <= upper))).astype(float)
        
        # Exclude instances with 0 variation in mask natively
        if np.std(mask_bins) == 0:
            continue
            
        res = mantel(D_e, mask_bins, permutations=permutations)
        class_dist = (lower + upper) / 2.0
        
        results.append({
            "class_dist": class_dist,
            "mantel_r": res["statistic"],
            "p_value": res["significance"]
        })
        
    return pd.DataFrame(results)

def dispindmorisita(x: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
    """
    Morisita's Index of Dispersion.
    
    Mimics `vegan::dispindmorisita`. Mathematically calculates spatial overdispersion boundaries stably.
    """
    x_arr = np.asarray(x, dtype=float)
    is_df = isinstance(x, pd.DataFrame)
    
    n_plots = x_arr.shape[0]
    N = np.sum(x_arr, axis=0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        num = n_plots * np.sum(x_arr * (x_arr - 1), axis=0)
        den = N * (N - 1)
        Im = np.where(N > 1, num / den, np.nan)
        
    df_out = pd.DataFrame({"imor": Im})
    if is_df:
        df_out.index = x.columns
    return df_out

def nestednodf(x: Union[np.ndarray, pd.DataFrame], order: bool = True) -> dict:
    """
    Nestedness Metric based on Overlap and Decreasing Fill (NODF).
    
    Mimics `vegan::nestednodf`. Dynamically computes identical structural occurrence bounds 
    scaling fractions exactly to $100\%$ overlapping limits identifying formal nested spatial thresholds natively!
    
    Parameters
    ----------
    x : array-like
        Community occurrences explicitly mapped linearly.
    order : bool
        If True, ranks rows and columns explicitly mapping structural sorting boundaries mathematically!
        
    Returns
    -------
    dict
        "N.columns", "N.rows", and "NODF" totals mathematically!
    """
    x_inc = (np.asarray(x) > 0).astype(int)
    
    if order:
        row_sums = x_inc.sum(axis=1)
        r_idx = np.argsort(-row_sums, kind='mergesort')
        x_inc = x_inc[r_idx, :]
        
        col_sums = x_inc.sum(axis=0)
        c_idx = np.argsort(-col_sums, kind='mergesort')
        x_inc = x_inc[:, c_idx]
        
    r, c = x_inc.shape
    row_sums = x_inc.sum(axis=1)
    col_sums = x_inc.sum(axis=0)
    
    row_paired = []
    for i in range(r - 1):
        for j in range(i + 1, r):
            if row_sums[i] <= row_sums[j]:
                row_paired.append(0.0)
            else:
                overlap = np.sum(x_inc[i] & x_inc[j])
                val = 100.0 * overlap / row_sums[j] if row_sums[j] > 0 else 0.0
                row_paired.append(val)
                
    col_paired = []
    for i in range(c - 1):
        for j in range(i + 1, c):
            if col_sums[i] <= col_sums[j]:
                col_paired.append(0.0)
            else:
                overlap = np.sum(x_inc[:, i] & x_inc[:, j])
                val = 100.0 * overlap / col_sums[j] if col_sums[j] > 0 else 0.0
                col_paired.append(val)
                
    N_rows = np.mean(row_paired) if len(row_paired) > 0 else 0.0
    N_cols = np.mean(col_paired) if len(col_paired) > 0 else 0.0
    N_total = np.mean(row_paired + col_paired) if len(row_paired) + len(col_paired) > 0 else 0.0
    
    return {
        "N.columns": N_cols,
        "N.rows": N_rows,
        "NODF": N_total
    }




