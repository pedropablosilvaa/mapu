import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform
from mapu.ordination import cmdscale, metaMDS
from mapu.vegdist import vegdist

def test_cmdscale():
    # Simple coordinates
    pts = np.array([
        [0, 0],
        [1, 0],
        [0, 1]
    ])
    # Euclidean distances
    dist = pdist(pts, metric="euclidean")
    
    points, eig = cmdscale(dist, k=2)
    
    assert points.shape == (3, 2)
    # The sum of eigenvalues should equal the sum of variances...
    # Just check it returns valid output without error
    assert len(eig) == 3
    # First eigenvalue should be positive
    assert eig[0] > 0

def test_metaMDS():
    # Simulated community data
    np.random.seed(42)
    data = np.random.randint(0, 10, size=(10, 5))
    
    # Run NMDS
    coords = metaMDS(data, distance="bray", k=2, n_init=1)
    
    assert coords.shape == (10, 2)
