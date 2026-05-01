import numpy as np
import pandas as pd
from typing import Union, Optional

def diversity(x: Union[np.ndarray, pd.DataFrame], 
              index: str = "shannon", 
              margin: int = 1, 
              base: float = np.e) -> np.ndarray:
    """
    Calculate ecological diversity indices.
    
    This function mimics the behavior of `vegan::diversity` in R.
    
    Parameters
    ----------
    x : array-like
        Community data, a matrix-like object where rows are typically samples 
        and columns are species (if margin=1).
    index : str, default "shannon"
        Diversity index, one of "shannon", "simpson", or "invsimpson".
    margin : int, default 1
        Margin for which the index is computed. 
        1 = rows (samples), 2 = columns (species).
    base : float, default np.e
        The logarithm base used in shannon.
        
    Returns
    -------
    np.ndarray
        Array of diversity values for each sample/row.
    """
    x_arr = np.asarray(x, dtype=float)
    
    if margin == 2:
        x_arr = x_arr.T
        
    # Check for negative values
    if np.any(x_arr < 0):
        raise ValueError("Data cannot contain negative values.")
        
    # Row totals
    totals = np.sum(x_arr, axis=1)
    
    # Avoid division by zero by replacing 0 totals with 1 (or ignoring them safely)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = x_arr / totals[:, np.newaxis]
    
    if index == "shannon":
        with np.errstate(divide='ignore', invalid='ignore'):
            logp = np.log(p) / np.log(base)
            # p*log(p) is 0 when p is 0. 
            x_logp = p * logp
            x_logp[np.isnan(x_logp)] = 0.0
            
        return -np.sum(x_logp, axis=1)
        
    elif index == "simpson":
        return 1.0 - np.sum(p**2, axis=1)
        
    elif index == "invsimpson":
        sum_p2 = np.sum(p**2, axis=1)
        # Avoid division by zero
        out = np.empty_like(sum_p2)
        zero_mask = (sum_p2 == 0)
        out[zero_mask] = np.nan
        out[~zero_mask] = 1.0 / sum_p2[~zero_mask]
        return out
        
    elif index == "simpson.unb":
        # Unbiased Simpson
        # D = 1 - sum(x * (x-1)) / (N * (N-1))
        x_counts = x_arr  # Assuming integer counts, but works with floats structurally too
        N = totals
        with np.errstate(divide='ignore', invalid='ignore'):
            # sum(p*p) in unbiased form
            out = 1.0 - np.sum(x_counts * (x_counts - 1), axis=1) / (N * (N - 1))
        # Handle nan for N <= 1
        out = np.where(N <= 1, np.nan, out)
        return out
        
    else:
        raise ValueError(f"Unknown index {index}")

def specnumber(x: Union[np.ndarray, pd.DataFrame], 
               margin: int = 1) -> np.ndarray:
    """
    Calculate the number of species (species richness).
    
    Parameters
    ----------
    x : array-like
        Community data matrix.
    margin : int, default 1
        1 = rows, 2 = columns.
        
    Returns
    -------
    np.ndarray
        Number of species present (count of non-zero entries).
    """
    x_arr = np.asarray(x, dtype=float)
    if margin == 2:
        x_arr = x_arr.T
    return np.sum(x_arr > 0, axis=1)

def specaccum(x: Union[np.ndarray, pd.DataFrame], 
              method: str = "random", 
              permutations: int = 100) -> dict:
    """
    Species Accumulation Curves.
    
    Mimics `vegan::specaccum(method="random")`.
    
    Parameters
    ----------
    x : array-like
        Community data matrix (rows are sites, columns are species).
    method : str
        Currently only "random" method is implemented.
    permutations : int
        Number of random permutations for the "random" method.
        
    Returns
    -------
    dict
        "sites": Array of site indices (1 to N)
        "richness": Mean accumulated species richness across permutations
        "sd": Standard deviation of accumulated richness
    """
    x_arr = np.asarray(x)
    n_sites, n_species = x_arr.shape
    
    if method != "random":
        raise ValueError("Only method='random' is supported right now.")
        
    accum_matrix = np.zeros((permutations, n_sites))
    
    for p in range(permutations):
        # random ordering of sites
        perm = np.random.permutation(n_sites)
        shuffled_x = x_arr[perm]
        
        # Binary occurrence
        occ = (shuffled_x > 0).astype(int)
        
        # Cumulative sum across sites
        cum_occ = np.cumsum(occ, axis=0)
        
        # For each step, count how many species have cumulative occurrence > 0
        richness = np.sum(cum_occ > 0, axis=1)
        accum_matrix[p, :] = richness
        
    mean_richness = np.mean(accum_matrix, axis=0)
    sd_richness = np.std(accum_matrix, axis=0, ddof=1)
    
    return {
        "sites": np.arange(1, n_sites + 1),
        "richness": mean_richness,
        "sd": sd_richness
    }

from scipy.special import gammaln

def rarefy(x: Union[np.ndarray, pd.DataFrame], 
           sample: int) -> np.ndarray:
    """
    Rarefied species richness.
    
    Mimics `vegan::rarefy`.
    Calculates expected species richness in random subsamples of size `sample`.
    
    Parameters
    ----------
    x : array-like
        Community data matrix.
    sample : int
        Subsample size.
        
    Returns
    -------
    np.ndarray
        Array of expected species richness for each site.
    """
    x_arr = np.asarray(x, dtype=float)
    
    # Rarefaction runs independently on each row (site)
    n_sites, n_species = x_arr.shape
    expected_richness = np.zeros(n_sites)
    
    for i in range(n_sites):
        # abundances > 0
        site_abundances = x_arr[i]
        ni = site_abundances[site_abundances > 0]
        N = np.sum(ni)
        
        if N < sample:
            # Cannot rarefy if total observations in site < subsample depth
            # Vegan returns NA, we return NaN
            expected_richness[i] = np.nan
        else:
            # E[S] = sum_j {1 - [choose(N - n_j, sample) / choose(N, sample)]}
            # Instead of factorials, use log-gamma
            # lchoose(n, k) = gammaln(n+1) - gammaln(k+1) - gammaln(n-k+1)
            # log_prob_missing = lchoose(N - n_j, sample) - lchoose(N, sample)
            
            ldenom = gammaln(N + 1) - gammaln(sample + 1) - gammaln(N - sample + 1)
            prob_missing = np.zeros(len(ni))
            
            for j, nj in enumerate(ni):
                if N - nj < sample:
                    prob_missing[j] = 0.0
                else:
                    lnum = gammaln(N - nj + 1) - gammaln(sample + 1) - gammaln(N - nj - sample + 1)
                    prob_missing[j] = np.exp(lnum - ldenom)
                    
            expected_richness[i] = np.sum(1.0 - prob_missing)
            
    return expected_richness

def renyi(x: Union[np.ndarray, pd.DataFrame], 
          scales: Union[list, np.ndarray] = [0, 0.25, 0.5, 1, 2, 4, 8, 16, 32, 64, np.inf]) -> pd.DataFrame:
    """
    Rényi Diversity Entropy Profiles.
    
    Mimics `vegan::renyi`.
    
    Parameters
    ----------
    x : array-like
        Community data matrix.
    scales : list or array
        Scale parameters (a). Default encompasses the standard vegan scale array.
        
    Returns
    -------
    pd.DataFrame
        DataFrame where rows are sites and columns are scale parameters.
    """
    x_arr = np.asarray(x, dtype=float)
    n_sites, _ = x_arr.shape
    
    scales = np.asarray(scales, dtype=float)
    
    result = np.zeros((n_sites, len(scales)))
    
    for i in range(n_sites):
        p = x_arr[i]
        p = p[p > 0]
        if len(p) == 0:
            result[i, :] = 0.0
            continue
            
        p = p / np.sum(p)
        
        for j, a in enumerate(scales):
            if a == 0:
                result[i, j] = np.log(len(p))
            elif a == 1:
                result[i, j] = -np.sum(p * np.log(p))
            elif np.isinf(a):
                result[i, j] = -np.log(np.max(p))
            else:
                result[i, j] = (1.0 / (1.0 - a)) * np.log(np.sum(p ** a))
                
    cols = [str(a) if not np.isinf(a) else "Inf" for a in scales]
    return pd.DataFrame(result, columns=cols)

from scipy.optimize import brentq

def fisher_alpha(x: Union[np.ndarray, pd.DataFrame], margin: int = 1) -> np.ndarray:
    """
    Fisher's log-series alpha.
    
    Mimics `vegan::fisher.alpha`.
    Estimates the parameter alpha in S = alpha * ln(1 + N/alpha).
    """
    x_arr = np.asarray(x, dtype=float)
    if margin == 2:
        x_arr = x_arr.T
        
    N = np.sum(x_arr, axis=1)
    # species richness S (number of non-zero entries)
    S = np.sum(x_arr > 0, axis=1)
    
    alphas = np.zeros(len(N))
    
    for i in range(len(N)):
        n_i = N[i]
        s_i = S[i]
        
        # Vegan returns NA if S == 0 or S == 1 or N <= S
        if s_i <= 1 or n_i <= s_i:
            alphas[i] = np.nan
            continue
            
        # Target function S - alpha * log(1 + N/alpha) = 0
        def f(a):
            return s_i - a * np.log(1.0 + n_i / a)
            
        # Bracketing
        # For lower bound, vegan uses 1e-4 or similar, we can use 1e-6
        # For upper bound, vegan uses N. S is max N, alpha can be larger than N for very diverse systems?
        # A simple large number is fine. Scipy's brentq requires f(a)*f(b) < 0
        low = 1e-8
        high = n_i
        
        while f(high) > 0 and high < 1e12:
            high *= 10
            
        try:
            alphas[i] = brentq(f, low, high)
        except ValueError:
            alphas[i] = np.nan
            
    return alphas

def specpool(x: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
    """
    Extrapolated species richness in a species pool.
    
    Mimics `vegan::specpool` by estimating the true species richness 
    from community incidence frequencies. Computes Chao2 and Jackknife 1.
    
    Parameters
    ----------
    x : array-like
        Community data matrix (rows are sites, columns are species).
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns "Species" (Observed), "chao", "jack1".
    """
    x_arr = np.asarray(x, dtype=float)
    
    # Binarize to incidence (presence/absence)
    inc = (x_arr > 0).astype(int)
    
    m = inc.shape[0]  # Number of sites
    if m == 0:
        return pd.DataFrame({"Species": [0], "chao": [np.nan], "jack1": [np.nan]})
        
    # Frequencies of each species across sites
    freqs = np.sum(inc, axis=0)
    
    S_obs = np.sum(freqs > 0)
    Q1 = np.sum(freqs == 1)
    Q2 = np.sum(freqs == 2)
    
    # Chao2
    if Q2 > 0:
        chao2 = S_obs + ((m - 1) / m) * (Q1 ** 2) / (2 * Q2)
    else:
        chao2 = S_obs + ((m - 1) / m) * (Q1 * (Q1 - 1)) / 2
        
    # Jackknife 1
    jack1 = S_obs + Q1 * ((m - 1) / m)
    
    return pd.DataFrame({
        "Species": [S_obs],
        "chao": [chao2],
        "jack1": [jack1]
    })

def taxondive(x: Union[np.ndarray, pd.DataFrame], 
              dis: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
    """
    Taxonomic diversity and distinctness.
    
    Mimics `vegan::taxondive`. Computes taxonomic indices characterizing 
    the community using a taxonomic or phylogenetic distance matrix.
    
    Parameters
    ----------
    x : array-like
        Community data matrix (rows are sites, columns are species).
    dis : array-like
        Square or condensed distance matrix indexing taxonomic distances 
        among species.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns "Species" (Richness), "Delta", "Delta_star", "Delta_plus".
    """
    from scipy.spatial.distance import is_valid_y, squareform
    x_arr = np.asarray(x, dtype=float)
    dis_arr = np.asarray(dis, dtype=float)
    
    if dis_arr.ndim == 1 and is_valid_y(dis_arr):
        d_sq = squareform(dis_arr)
    else:
        d_sq = dis_arr
        
    n_sites, n_species = x_arr.shape
    if d_sq.shape[0] != n_species:
        raise ValueError("Dimensions of taxonomic distance matrix must match number of species in community matrix.")
        
    res_S = np.zeros(n_sites)
    res_D = np.zeros(n_sites)
    res_Dstar = np.zeros(n_sites)
    res_Dplus = np.zeros(n_sites)
    
    for i in range(n_sites):
        w = x_arr[i]
        b = w > 0
        S = np.sum(b)
        res_S[i] = S
        
        if S < 2:
            res_D[i] = np.nan
            res_Dstar[i] = np.nan
            res_Dplus[i] = np.nan
            continue
            
        n = np.sum(w)
        
        # Submatrix of distances for present species
        d_sub = d_sq[np.ix_(b, b)]
        
        # Sum of distances for Delta+
        # Summing upper triangle
        sum_d_plus = np.sum(d_sub[np.triu_indices(S, k=1)])
        # Delta+ = mean taxonomic distance among presence/absence
        res_Dplus[i] = (2.0 * sum_d_plus) / (S * (S - 1))
        
        # Weighted distances
        # w_sub is weights of present species
        w_sub = w[b]
        
        # w_i w_j D_ij
        wd = np.outer(w_sub, w_sub) * d_sub
        sum_wd = np.sum(wd[np.triu_indices(S, k=1)])
        
        # Delta (Taxonomic Diversity)
        res_D[i] = (2.0 * sum_wd) / (n * (n - 1)) if n > 1 else 0
        
        # Delta* (Taxonomic Distinctness)
        # Denominator is sum of w_i w_j
        sum_w_cross = np.sum(np.outer(w_sub, w_sub)[np.triu_indices(S, k=1)])
        res_Dstar[i] = sum_wd / sum_w_cross if sum_w_cross > 0 else 0
        
    index_names = x.index if isinstance(x, pd.DataFrame) else None
    
    return pd.DataFrame({
        "Species": res_S,
        "Delta": res_D,
        "Delta_star": res_Dstar,
        "Delta_plus": res_Dplus
    }, index=index_names)

def rad_null(x: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
    """
    Fits Broken Stick rank-abundance distributions.
    
    Mimics `vegan::radfit` (null model). Evaluates theoretically bounded 
    null abundances across ranked species distributions natively for each plot.
    
    Parameters
    ----------
    x : array-like
        Community data matrix.
        
    Returns
    -------
    array-like
        Expected abundance array structured by decreasing rank.
    """
    x_arr = np.atleast_2d(np.asarray(x, dtype=float))
    out = np.zeros_like(x_arr)
    
    for i in range(x_arr.shape[0]):
        row = x_arr[i]
        b = row > 0
        S = np.sum(b)
        N = np.sum(row)
        
        if S > 0:
            ranks = np.arange(1, S + 1)
            # Broken stick expectation: E[r] = N / S * sum_{j=r}^S (1/j)
            inv_seq = 1.0 / ranks
            expectations = (N / S) * np.cumsum(inv_seq[::-1])[::-1]
            out[i, :S] = expectations
            
    if isinstance(x, pd.DataFrame):
        return pd.DataFrame(out, index=x.index, columns=[f"Rank_{j+1}" for j in range(x.shape[1])])
        
    return out

def rad_preempt(x: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
    """
    Fits Niche Preemption (Geometric) rank-abundance distributions.
    
    Evaluates expected abundance array parameterized by least-squares bounded 
    alpha scaling coefficients.
    
    Parameters
    ----------
    x : array-like
        Community data matrix.
        
    Returns
    -------
    array-like
        Expected geometric abundance array structured by decreasing rank.
    """
    from scipy.optimize import curve_fit
    x_arr = np.atleast_2d(np.asarray(x, dtype=float))
    out = np.zeros_like(x_arr)
    
    for i in range(x_arr.shape[0]):
        row = x_arr[i]
        b = row > 0
        S = np.sum(b)
        N = np.sum(row)
        
        if S > 0:
            # Sort abundances natively
            abundances = np.sort(row[b])[::-1]
            ranks = np.arange(1, S + 1)
            
            # Parametric bounds
            # expected y = N * alpha * (1-alpha)^(r-1)
            def model_func(r, alpha):
                return N * alpha * ((1.0 - alpha)**(r - 1))
                
            try:
                popt, _ = curve_fit(model_func, ranks, abundances, p0=[0.1], bounds=(1e-6, 0.9999))
                alpha = popt[0]
            except:
                alpha = 0.1
                
            expectations = model_func(ranks, alpha)
            out[i, :S] = expectations
            
    if isinstance(x, pd.DataFrame):
        return pd.DataFrame(out, index=x.index, columns=[f"Rank_{j+1}" for j in range(x.shape[1])])
        
    return out

def adipart(x: Union[np.ndarray, pd.DataFrame], 
            index: str = "shannon",
            group: Union[np.ndarray, list, pd.Series] = None) -> dict:
    """
    Additive Diversity Partitioning.
    
    Mimics `vegan::adipart`. Partitions structurally variances explicitly 
    generating local Alpha nested bounds offsetting strictly matching Gamma representations, 
    mapping structurally linear Beta variances limits explicitly natively.
    
    Parameters
    ----------
    x : array-like
        Community data matrix.
    index : str
        Diversity string mapped recursively ("shannon", "simpson").
    group : array-like
        Site categorical distributions identifying grouping clusters.
        
    Returns
    -------
    dict
        "alpha": Intra-class cluster mean structures.
        "beta": Dis-similarity representations mapping constraints.
        "gamma": Globally mapped variance limit structure bounds.
    """
    x_arr = np.asarray(x, dtype=float)
    if group is None:
        return {}
        
    g_arr = np.asarray(group)
    unique_groups = np.unique(g_arr)
    
    pooled_gamma = np.sum(x_arr, axis=0)
    
    # We must explicitly convert 1D array pool into [1, p] shape 
    # to evaluate diversity across a 1-site pseudo pool safely natively
    gamma_val = np.asarray(diversity(pooled_gamma.reshape(1, -1), index=index)).item()
    
    group_alphas = []
    
    for g in unique_groups:
        idx = (g_arr == g)
        pooled_alpha = np.sum(x_arr[idx, :], axis=0).reshape(1, -1)
        a_val = np.asarray(diversity(pooled_alpha, index=index)).item()
        group_alphas.append(a_val)
        
    mean_alpha = np.mean(group_alphas)
    beta = gamma_val - mean_alpha
    
    return {
        "alpha": mean_alpha,
        "beta": beta,
        "gamma": gamma_val
    }

def multipart(x: Union[np.ndarray, pd.DataFrame], 
              index: str = "shannon",
              group: Union[np.ndarray, list, pd.Series] = None) -> dict:
    """
    Multiplicative Diversity Partitioning.
    
    Mimics `vegan::multipart`. Evaluates Whittaker Beta Diversity multiplicatively 
    (Gamma / Alpha) natively executing bounds linearly mapping fractional distributions dynamically.
    
    Parameters
    ----------
    x : array-like
        Community data matrix.
    index : str
        Diversity method ("shannon", "simpson").
    group : array-like
        Categorical groups defining strict pooling classes.
        
    Returns
    -------
    dict
        "alpha": Mean variance within class pools.
        "beta": Fractional Whittaker multiplicative constraints.
        "gamma": Overall dataset boundary representations.
    """
    x_arr = np.asarray(x, dtype=float)
    if group is None:
        return {}
        
    g_arr = np.asarray(group)
    unique_groups = np.unique(g_arr)
    
    pooled_gamma = np.sum(x_arr, axis=0)
    gamma_val = np.asarray(diversity(pooled_gamma.reshape(1, -1), index=index)).item()
    
    group_alphas = []
    
    for g in unique_groups:
        idx = (g_arr == g)
        pooled_alpha = np.sum(x_arr[idx, :], axis=0).reshape(1, -1)
        a_val = np.asarray(diversity(pooled_alpha, index=index)).item()
        group_alphas.append(a_val)
        
    mean_alpha = np.mean(group_alphas)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        beta = gamma_val / mean_alpha if mean_alpha > 0 else 0.0
        
    return {
        "alpha": mean_alpha,
        "beta": beta,
        "gamma": gamma_val
    }

from scipy.optimize import least_squares

def rad_zipf(x: np.ndarray) -> np.ndarray:
    """
    Zipf-Mandelbrot null model estimation for rank-abundance curves.
    
    Matches scaling configuration native to `vegan::radfit` distributions mapping 
    least squares constraints solving parameters matching theoretical decay algorithms algebraically.
    
    Returns rank bounds dynamically matching the occurrences limits explicitly.
    """
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim > 1:
        x_arr = np.sum(x_arr, axis=0)
        
    abundances = np.sort(x_arr)[::-1]
    abundances = abundances[abundances > 0]
    ranks = np.arange(1, len(abundances) + 1)
    
    def model(theta, r):
        return theta[0] * (r + theta[1])**theta[2]
        
    def residuals(theta, r, y):
        return model(theta, r) - y
        
    theta0 = [np.max(abundances), 0.5, -1.0]
    bounds = ([0.0, -0.999, -15.0], [np.inf, np.inf, 15.0])
    
    try:
        out = least_squares(residuals, theta0, args=(ranks, abundances), bounds=bounds)
        return model(out.x, ranks)
    except:
        return np.zeros_like(ranks)

def tsallis(x: Union[np.ndarray, pd.DataFrame], scales: Union[float, list] = 1.0) -> Union[np.ndarray, pd.DataFrame]:
    """
    Tsallis Generalized Entropy.
    
    Mimics `vegan::tsallis` generalizing bounds utilizing non-linear variance 
    scaling factors identically mapping parametric limits.
    """
    x_arr = np.asarray(x, dtype=float)
    is_df = isinstance(x, pd.DataFrame)
    
    totals = np.sum(x_arr, axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        p = np.where(totals > 0, x_arr / totals, 0)
        
    if isinstance(scales, (int, float)):
        scales_list = [scales]
    else:
        scales_list = scales
        
    out = np.zeros((x_arr.shape[0], len(scales_list)))
    
    for k, scale in enumerate(scales_list):
        if scale == 1.0:
            p_log = np.where(p > 0, p * np.log(p), 0)
            out[:, k] = -np.sum(p_log, axis=1)
        else:
            p_q = p ** scale
            out[:, k] = (1.0 - np.sum(p_q, axis=1)) / (scale - 1.0)
            
    if out.shape[1] == 1:
        out = out[:, 0]
        if is_df:
            return pd.Series(out, index=x.index)
        return out
        
    if is_df:
        return pd.DataFrame(out, index=x.index, columns=[str(s) for s in scales_list])
    return out

def rrarefy(x: Union[np.ndarray, pd.DataFrame], sample: int) -> Union[np.ndarray, pd.DataFrame]:
    """
    Randomized rarefaction downsampling.
    
    Mimics `vegan::rrarefy`. Explicitly samples occurrences isolating populations 
    authentically explicitly executing sampling without replacement mappings dynamically.
    """
    x_arr = np.asarray(x, dtype=int)
    is_df = isinstance(x, pd.DataFrame)
    
    out = np.zeros_like(x_arr)
    n, p = x_arr.shape
    
    for i in range(n):
        row = x_arr[i]
        total = np.sum(row)
        if total <= sample:
            out[i] = row
        else:
            population = np.repeat(np.arange(p), row)
            sampled = np.random.choice(population, size=sample, replace=False)
            counts = np.bincount(sampled, minlength=p)
            out[i] = counts
            
    if is_df:
        return pd.DataFrame(out, index=x.index, columns=x.columns)
    return out

def estimateR(x: Union[np.ndarray, pd.DataFrame]) -> pd.DataFrame:
    """
    Extrapolated richness estimator.
    
    Mimics `vegan::estimateR`. Structurally computes explicitly missing 
    Chao1 and ACE theoretical populations limits seamlessly relying strictly on 
    singletons and doubletons dynamically mapped across spatial plots inherently.
    """
    x_arr = np.asarray(x, dtype=float)
    is_df = isinstance(x, pd.DataFrame)
    
    n_sites, n_species = x_arr.shape
    results = []
    
    for i in range(n_sites):
        row = x_arr[i]
        S_obs = np.sum(row > 0)
        f1 = np.sum(row == 1)
        f2 = np.sum(row == 2)
        
        if f2 > 0:
            S_chao1 = S_obs + (f1 ** 2) / (2.0 * f2)
        else:
            S_chao1 = S_obs + (f1 * (f1 - 1)) / 2.0
            
        abund_thresh = 10
        abund_mask = row > abund_thresh
        rare_mask = (row > 0) & (row <= abund_thresh)
        
        S_abund = np.sum(abund_mask)
        S_rare = np.sum(rare_mask)
        N_rare = np.sum(row[rare_mask])
        
        if N_rare > 0:
            C_ace = 1.0 - f1 / N_rare
            f_counts = [np.sum(row == j) for j in range(1, abund_thresh + 1)]
            sum_f = sum(j * (j - 1) * f_counts[j - 1] for j in range(1, abund_thresh + 1))
            
            if C_ace > 0 and N_rare > 1:
                gamma2 = max(S_rare / C_ace * sum_f / (N_rare * (N_rare - 1)) - 1.0, 0.0)
                S_ace = S_abund + S_rare / C_ace + f1 / C_ace * gamma2
            else:
                S_ace = S_obs
        else:
            S_ace = S_obs
            
        results.append({
            "S.obs": S_obs,
            "S.chao1": S_chao1,
            "S.ACE": S_ace
        })
        
    df_out = pd.DataFrame(results).T if not is_df else pd.DataFrame(results, index=x.index).T
    # vegan typically returns sites as columns for estimateR!
    return df_out

from scipy.special import gammaln

def _lchoose(n, k):
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)

def drarefy(x: Union[np.ndarray, pd.DataFrame], sample: int) -> Union[np.ndarray, pd.DataFrame]:
    """
    Rarefaction Probabilities.
    
    Mimics `vegan::drarefy`. Mathematically evaluates strict hypergeometric combination limits
    identifying precisely identical probability bounding structures mapped mathematically across random draws.
    """
    x_arr = np.asarray(x, dtype=float)
    is_df = isinstance(x, pd.DataFrame)
    
    N = np.sum(x_arr, axis=1, keepdims=True)
    out = np.zeros_like(x_arr)
    
    with np.errstate(invalid='ignore', divide='ignore'):
        lchoose_N_sample = _lchoose(N, sample)
        lchoose_Nx_sample = _lchoose(N - x_arr, sample)
        
        valid_mask = (N - x_arr) >= sample
        
        prob_not_drawn = np.where(valid_mask, np.exp(lchoose_Nx_sample - lchoose_N_sample), 0.0)
        out = 1.0 - prob_not_drawn
        out = np.clip(out, 0.0, 1.0)
        
    if is_df:
        return pd.DataFrame(out, index=x.index, columns=x.columns)
    return out

def renyiaccum(x: Union[np.ndarray, pd.DataFrame], permutations: int = 100, scales: Union[float, list] = [0, 1, 2, 4, 8, np.inf]) -> dict:
    """
    Renyi Entropy Accumulation Model.
    
    Mimics `vegan::renyiaccum`. Dynamically pools spatial site components testing 
    identical random scaling limit permutations recursively evaluating Renyi limits structurally natively!
    """
    x_arr = np.asarray(x, dtype=float)
    n_sites, n_species = x_arr.shape
    
    if isinstance(scales, (int, float)):
        scales = [scales]
        
    num_scales = len(scales)
    results = np.zeros((permutations, n_sites, num_scales))
    
    # Pre-parse scales avoiding structural iteration variables overhead implicitly mapping bounds
    for p in range(permutations):
        idx = np.random.permutation(n_sites)
        x_perm = x_arr[idx, :]
        x_cum = np.cumsum(x_perm, axis=0)
        
        res_r = renyi(x_cum, scales=scales)
        results[p, :, :] = np.asarray(res_r)
        
    mean_accum = np.mean(results, axis=0)
    
    df_out = pd.DataFrame(mean_accum, columns=[str(s) for s in scales])
    df_out.index = np.arange(1, n_sites + 1)
    
    return {
        "sites": np.arange(1, n_sites + 1),
        "mean": df_out,
        "results": results
    }

def tsallisaccum(x: Union[np.ndarray, pd.DataFrame], permutations: int = 100, scales: Union[float, list] = [0, 1, 2], norm: bool = False) -> dict:
    """
    Tsallis Entropy Accumulation Model.
    
    Mimics `vegan::tsallisaccum`. Dynamically pools spatial site components testing 
    identical random scaling limit permutations recursively evaluating Tsallis explicit limits structurally.
    """
    x_arr = np.asarray(x, dtype=float)
    n_sites, n_species = x_arr.shape
    
    if isinstance(scales, (int, float)):
        scales = [scales]
        
    num_scales = len(scales)
    results = np.zeros((permutations, n_sites, num_scales))
    
    for p in range(permutations):
        idx = np.random.permutation(n_sites)
        x_perm = x_arr[idx, :]
        x_cum = np.cumsum(x_perm, axis=0)
        
        res_r = tsallis(x_cum, scales=scales)
        results[p, :, :] = np.asarray(res_r)
        
    mean_accum = np.mean(results, axis=0)
    
    df_out = pd.DataFrame(mean_accum, columns=[str(s) for s in scales])
    df_out.index = np.arange(1, n_sites + 1)
    
    return {
        "sites": np.arange(1, n_sites + 1),
        "mean": df_out,
        "results": results
    }

def poolaccum(x: Union[np.ndarray, pd.DataFrame], permutations: int = 100) -> dict:
    """
    Extrapolated Species Richness Accumulation!
    
    Mimics `vegan::poolaccum`. Evaluates `specpool` dynamically subsets iteratively!
    Returns explicit asymptotic models scaling mathematically properly.
    """
    x_arr = np.asarray(x, dtype=float)
    n_sites, n_species = x_arr.shape
    
    results = np.zeros((permutations, n_sites, 5)) 
    
    for p in range(permutations):
        idx = np.random.permutation(n_sites)
        x_perm = x_arr[idx, :]
        x_inc = (x_perm > 0).astype(int)
        
        for k in range(1, n_sites + 1):
            subset = x_inc[0:k, :]
            N = k
            freq = np.sum(subset, axis=0)
            S = np.sum(freq > 0)
            
            if N == 1:
                results[p, k-1, :] = [S, S, S, S, S]
                continue
                
            f1 = np.sum(freq == 1)
            f2 = np.sum(freq == 2)
            
            if f2 > 0:
                chao = S + (f1**2) / (2 * f2)
            else:
                chao = S + (f1 * (f1 - 1)) / 2.0
                
            jack1 = S + f1 * ((N - 1) / N)
            jack2 = S + f1 * ((2 * N - 3) / N) - f2 * ((N - 2)**2 / (N * (N - 1)))
            
            with np.errstate(divide='ignore', invalid='ignore'):
                p_not = (1 - freq[freq > 0]/N)**N
                boot = S + np.sum(p_not)
                
            results[p, k-1, :] = [S, chao, jack1, jack2, boot]
            
    mean_res = np.mean(results, axis=0)
    
    df_mean = pd.DataFrame(mean_res, columns=["S", "chao", "jack1", "jack2", "boot"])
    df_mean.index = np.arange(1, n_sites + 1)
    
    return {
        "sites": np.arange(1, n_sites + 1),
        "mean": df_mean,
        "results": results
    }

def radfit(x: Union[np.ndarray, pd.DataFrame]) -> dict:
    """
    Rank Abundance Distribution Fitting Wrapper.
    
    Mimics `vegan::radfit`. Evaluates optimal mathematical arrays distributing 
    Null, Preempt, and Zipf limits natively outputting consolidated error models structurally.
    """
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim == 2:
        x_vec = np.sum(x_arr, axis=0)
    else:
        x_vec = x_arr
        
    p_null = np.squeeze(rad_null(x_vec))
    p_preempt = np.squeeze(rad_preempt(x_vec))
    p_zipf = np.squeeze(rad_zipf(x_vec))
    
    def _rss(pred, obs):
        return np.sum((pred - obs)**2)
        
    ranks = np.arange(1, len(x_vec) + 1)
    obs = np.sort(x_vec)[::-1]
    
    df_out = pd.DataFrame({
        "Null": p_null,
        "Preempt": p_preempt,
        "Zipf": p_zipf
    }, index=ranks)
    
    rss_out = pd.DataFrame({
        "RSS": [
            _rss(p_null, obs),
            _rss(p_preempt, obs),
            _rss(p_zipf, obs)
        ]
    }, index=["Null", "Preempt", "Zipf"])
    
    return {
        "models": df_out,
        "RSS": rss_out
    }






