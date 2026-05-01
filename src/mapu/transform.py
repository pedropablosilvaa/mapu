import numpy as np
import pandas as pd
from typing import Union

def decostand(x: Union[np.ndarray, pd.DataFrame], 
              method: str, 
              margin: int = 1,
              logbase: float = 2.0) -> Union[np.ndarray, pd.DataFrame]:
    """
    Standardization methods for community ecology.
    
    Mimics `vegan::decostand`.
    
    Parameters
    ----------
    x : array-like
        Community data matrix.
    method : str
        Standardization method: "total", "max", "pa", "hellinger", "normalize", "log", "freq".
    margin : int, default 1
        1 = rows (sites), 2 = columns (species).
    logbase : float, default 2.0
        Logarithm base for method="log". Default matches vegan (2.0).
        
    Returns
    -------
    Transformed array or DataFrame (matches input type if possible).
    """
    is_df = isinstance(x, pd.DataFrame)
    x_arr = np.asarray(x, dtype=float)
    
    if margin == 2:
        x_arr = x_arr.T
        
    out = np.empty_like(x_arr)
    
    if method == "total":
        totals = np.sum(x_arr, axis=1, keepdims=True)
        # avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            out = np.where(totals == 0, 0, x_arr / totals)
            
    elif method == "max":
        max_vals = np.max(x_arr, axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            out = np.where(max_vals == 0, 0, x_arr / max_vals)
            
    elif method == "pa":
        out = (x_arr > 0).astype(float)
        
    elif method == "hellinger":
        totals = np.sum(x_arr, axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            rel = np.where(totals == 0, 0, x_arr / totals)
        # Cannot take sqrt of negative numbers, but abundances should be >=0
        if np.any(rel < 0):
            raise ValueError("Data contains negative values; hellinger requires non-negative data.")
        out = np.sqrt(rel)
        
    elif method == "chi.square":
        # vegan chi.square margin calculation
        row_sums = np.sum(x_arr, axis=1, keepdims=True)
        col_sums = np.sum(x_arr, axis=0, keepdims=True)
        total_sum = np.sum(x_arr)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            # divisor is rowSums * sqrt(colSums)
            expected = row_sums * np.sqrt(col_sums)
            out = np.where(expected == 0, 0, x_arr / expected * np.sqrt(total_sum))
            
    elif method == "normalize":
        ss = np.sqrt(np.sum(x_arr**2, axis=1, keepdims=True))
        with np.errstate(divide='ignore', invalid='ignore'):
            out = np.where(ss == 0, 0, x_arr / ss)
            
    elif method == "log":
        # Vegan's log: x = log_b(x) + 1 for x > 0, 0 otherwise. Vegan default base is 2
        with np.errstate(divide='ignore', invalid='ignore'):
            if logbase == np.e:
                log_x = np.log(x_arr)
            else:
                log_x = np.log(x_arr) / np.log(logbase)
            out = np.where(x_arr > 0, log_x + 1, 0)
        
    elif method == "freq":
        # frequency among species: divide row by row total
        totals = np.sum(x_arr, axis=1, keepdims=True)
        with np.errstate(divide='ignore', invalid='ignore'):
            out = np.where(totals == 0, 0, x_arr / totals)
            
    else:
        raise ValueError(f"Unknown decostand method: {method}")
        
    if margin == 2:
        out = out.T
        
    if is_df:
        return pd.DataFrame(out, index=x.index, columns=x.columns)
    return out

def wisconsin(x: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
    """
    Wisconsin double standardization.
    
    Mimics `vegan::wisconsin`.
    Divides species by their maxima, and sites by their totals.
    
    Parameters
    ----------
    x : array-like
        Community data matrix.
        
    Returns
    -------
    Transformed array or DataFrame.
    """
    is_df = isinstance(x, pd.DataFrame)
    x_arr = np.asarray(x, dtype=float)
    
    # 1. Species divided by maxima
    max_vals = np.max(x_arr, axis=0, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        step1 = np.where(max_vals == 0, 0, x_arr / max_vals)
        
    # 2. Sites divided by totals
    totals = np.sum(step1, axis=1, keepdims=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(totals == 0, 0, step1 / totals)
        
    if is_df:
        return pd.DataFrame(out, index=x.index, columns=x.columns)
    return out

def beals(x: Union[np.ndarray, pd.DataFrame]) -> Union[np.ndarray, pd.DataFrame]:
    """
    Beals Smoothing.
    
    Mimics `vegan::beals`. Instead of raw abundances or incidences, replaces 
    components strictly with probabilities estimating occurrences strictly via 
    global co-occurrence structural mapping loops natively.
    
    Parameters
    ----------
    x : array-like
        Community data matrix.
        
    Returns
    -------
    array-like
        Probability bounding matrix structuring limits mathematically mapped.
    """
    x_arr = np.asarray(x, dtype=float)
    x_inc = (x_arr > 0).astype(float)
    
    S = np.sum(x_inc, axis=1, keepdims=True)
    N = np.sum(x_inc, axis=0, keepdims=True)
    
    # Co-occurrence matrix
    M = x_inc.T @ x_inc
    np.fill_diagonal(M, 0.0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        weights = np.where(N > 0, x_inc / N, 0)
        
    p_matrix = weights @ M
    
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(S > 0, p_matrix / S, 0.0)
        
    if isinstance(x, pd.DataFrame):
        return pd.DataFrame(out, index=x.index, columns=x.columns)
    return out

def dispweight(x: Union[np.ndarray, pd.DataFrame]) -> dict:
    """
    Dispersion Weighting.
    
    Mimics `vegan::dispweight`. Mathematically down-weights strongly over-dispersed 
    counts structuring variance-to-mean bounds dynamically across columns.
    
    Parameters
    ----------
    x : array-like
        Community data matrix.
        
    Returns
    -------
    dict
        "x": The transformed counts structuring strictly over dispersion bounds.
        "D": The calculated index of dispersion natively mapped across columns.
    """
    x_arr = np.asarray(x, dtype=float)
    is_df = isinstance(x, pd.DataFrame)
    
    # Calculate Mean and Sample Variance directly
    means = np.mean(x_arr, axis=0)
    var_unbiased = np.var(x_arr, axis=0, ddof=1)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        D = np.where(means > 0, var_unbiased / means, 1.0)
        
    # Weights constrain heavily inflated occurrences. D <= 1 implies unadjusted normal scaling natively.
    w = np.maximum(D, 1.0)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where(w > 0, x_arr / w, 0.0)
        
    if is_df:
        out = pd.DataFrame(out, index=x.index, columns=x.columns)
        D = pd.Series(D, index=x.columns)
        
    return {"x": out, "D": D}

def coverscale(x: Union[np.ndarray, pd.DataFrame], scale: str = "braun.blanquet") -> Union[np.ndarray, pd.DataFrame]:
    """
    Cover Scale Conversion.
    
    Mimics `vegan::coverscale`. Calculates formal mid-point percentage structures 
    dynamically substituting classical botanical ordinal ranks linearly identically.
    
    Parameters
    ----------
    x : array-like
        Ordinal array integers mappings.
    scale : str
        Dictionary substitution targeting ("braun.blanquet", "domin").
        
    Returns
    -------
    array-like
        Parsed quantitative arrays mapped natively!
    """
    x_arr = np.asarray(x, dtype=float)
    is_df = isinstance(x, pd.DataFrame)
    
    out = np.copy(x_arr)
    
    if scale.lower() == "braun.blanquet":
        lookup = {0: 0.0, 1: 1.0, 2: 15.0, 3: 37.5, 4: 62.5, 5: 87.5}
    elif scale.lower() == "domin":
        lookup = {0: 0.0, 1: 1.0, 2: 2.0, 3: 4.0, 4: 10.0, 5: 20.0, 6: 29.5, 7: 41.5, 8: 62.5, 9: 83.5, 10: 95.5}
    else:
        raise ValueError("scale must be 'braun.blanquet' or 'domin'")
        
    for k, v in lookup.items():
        out[x_arr == k] = v
        
    if is_df:
        return pd.DataFrame(out, index=x.index, columns=x.columns)
    return out

def make_cepnames(names: Union[list, np.ndarray, pd.Series]) -> list:
    """
    Abbreviates species nomenclature.
    
    Mimics `vegan::make.cepnames`. Condenses taxonomic bounds extracting 
    exactly 4 letters from the genus and 4 letters from the species identifying 
    abbreviated graphical variables effortlessly compactly natively.
    """
    def _abbr(s):
        parts = str(s).strip().split()
        if len(parts) == 1:
            return parts[0][:8]
        elif len(parts) >= 2:
            return parts[0][:4] + parts[1][:4]
        return ""
        
    return [_abbr(name) for name in names]
