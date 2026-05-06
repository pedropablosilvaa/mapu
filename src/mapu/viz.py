import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import ConvexHull
from matplotlib.patches import Ellipse
from typing import Union, Optional, List, Dict

def ordiplot(points: np.ndarray, 
             groups: Optional[Union[List, pd.Series, np.ndarray]] = None,
             ax: Optional[plt.Axes] = None,
             title: str = "Ordination Plot",
             cmap: str = "tab10",
             **kwargs) -> plt.Axes:
    """
    Base plot for ordination results (NMDS, PCA, RDA, etc.).
    Mimics `vegan::ordiplot`.
    
    Parameters
    ----------
    points : np.ndarray
        Coordinates of the points.
    groups : array-like, optional
        Grouping categories for coloring.
    ax : plt.Axes, optional
        Matplotlib axes to plot on.
    title : str
        Title of the plot.
    cmap : str
        Seaborn palette name for coloring groups.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    points = np.asarray(points)
    if points.shape[1] < 2:
        raise ValueError("points must have at least 2 dimensions")
        
    x, y = points[:, 0], points[:, 1]
    
    if groups is not None:
        sns.scatterplot(x=x, y=y, hue=groups, palette=cmap, ax=ax, **kwargs)
    else:
        ax.scatter(x, y, c='black', **kwargs)
        
    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    
    return ax

def ordihull(points: np.ndarray, 
             groups: Union[List, pd.Series, np.ndarray],
             ax: Optional[plt.Axes] = None,
             alpha: float = 0.2,
             cmap: str = "tab10",
             **kwargs) -> plt.Axes:
    """
    Add convex hulls to an ordination plot.
    Mimics `vegan::ordihull`.
    """
    if ax is None:
        ax = plt.gca()
        
    points = np.asarray(points)
    groups = np.asarray(groups)
    
    unique_groups = np.unique(groups)
    colors = sns.color_palette(cmap, len(unique_groups))
    
    for i, g in enumerate(unique_groups):
        idx = np.where(groups == g)[0]
        pts = points[idx, :2]
        
        if len(pts) >= 3:
            hull = ConvexHull(pts)
            hull_pts = pts[hull.vertices, :]
            poly = plt.Polygon(hull_pts, closed=True, fill=True, alpha=alpha, color=colors[i], **kwargs)
            ax.add_patch(poly)
            # Add edge
            poly_edge = plt.Polygon(hull_pts, closed=True, fill=False, color=colors[i])
            ax.add_patch(poly_edge)
            
    return ax

def ordiellipse(points: np.ndarray, 
                groups: Union[List, pd.Series, np.ndarray],
                ax: Optional[plt.Axes] = None,
                alpha: float = 0.2,
                cmap: str = "tab10",
                kind: str = "sd",
                **kwargs) -> plt.Axes:
    """
    Add standard deviation/error ellipses to an ordination plot.
    Mimics `vegan::ordiellipse`.
    """
    if ax is None:
        ax = plt.gca()
        
    points = np.asarray(points)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    colors = sns.color_palette(cmap, len(unique_groups))
    
    for i, g in enumerate(unique_groups):
        idx = np.where(groups == g)[0]
        pts = points[idx, :2]
        
        if len(pts) >= 3:
            cov = np.cov(pts, rowvar=False)
            val, vec = np.linalg.eigh(cov)
            
            # Sort eigenvalues
            order = val.argsort()[::-1]
            val, vec = val[order], vec[:, order]
            
            theta = np.degrees(np.arctan2(*vec[:, 0][::-1]))
            
            # Width and height (2 standard deviations)
            width, height = 2 * np.sqrt(val)
            
            ellipse = Ellipse(xy=np.mean(pts, axis=0), width=width, height=height, 
                              angle=theta, fill=True, alpha=alpha, color=colors[i], **kwargs)
            ax.add_patch(ellipse)
            ellipse_edge = Ellipse(xy=np.mean(pts, axis=0), width=width, height=height, 
                                   angle=theta, fill=False, color=colors[i])
            ax.add_patch(ellipse_edge)
            
    return ax

def ordispider(points: np.ndarray, 
               groups: Union[List, pd.Series, np.ndarray],
               ax: Optional[plt.Axes] = None,
               cmap: str = "tab10",
               **kwargs) -> plt.Axes:
    """
    Add spider webs (lines from centroid to points) to an ordination plot.
    Mimics `vegan::ordispider`.
    """
    if ax is None:
        ax = plt.gca()
        
    points = np.asarray(points)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    colors = sns.color_palette(cmap, len(unique_groups))
    
    for i, g in enumerate(unique_groups):
        idx = np.where(groups == g)[0]
        pts = points[idx, :2]
        
        if len(pts) > 0:
            centroid = np.mean(pts, axis=0)
            for pt in pts:
                ax.plot([centroid[0], pt[0]], [centroid[1], pt[1]], color=colors[i], **kwargs)
                
    return ax

def plot_specaccum(spec_result: pd.DataFrame, ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
    """
    Plot species accumulation curves with standard deviation.
    Mimics `plot.specaccum` in vegan.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    sites = spec_result["sites"]
    richness = spec_result["richness"]
    sd = spec_result.get("sd", None)
    
    ax.plot(sites, richness, color="black", label="Richness", **kwargs)
    
    if sd is not None:
        ax.fill_between(sites, richness - sd, richness + sd, alpha=0.2, color="blue", label="±1 SD")
        
    ax.set_xlabel("Number of Sites")
    ax.set_ylabel("Species Richness")
    ax.set_title("Species Accumulation Curve")
    ax.legend()
    
    return ax

def plot_rad(rad_result: pd.DataFrame, ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
    """
    Plot rank-abundance distributions for different models.
    Mimics `plot.radfit` in vegan.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
        
    if "Rank" not in rad_result.columns or "Observed" not in rad_result.columns:
        raise ValueError("rad_result must contain 'Rank' and 'Observed' columns")
        
    rank = rad_result["Rank"]
    
    ax.scatter(rank, rad_result["Observed"], color="black", label="Observed", zorder=5)
    
    # 5 standard models max in radfit usually
    colors = sns.color_palette("Set1", max(1, len(rad_result.columns) - 2))
    c_idx = 0
    
    for col in rad_result.columns:
        if col not in ["Rank", "Observed"]:
            ax.plot(rank, rad_result[col], label=col, color=colors[c_idx], **kwargs)
            c_idx += 1
            
    ax.set_yscale("log")
    ax.set_xlabel("Rank")
    ax.set_ylabel("Abundance")
    ax.set_title("Rank-Abundance Distribution")
    ax.legend()
    
    return ax
