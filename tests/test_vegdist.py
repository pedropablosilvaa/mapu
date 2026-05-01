import numpy as np
import pytest
from scipy.spatial.distance import squareform
from mapu.vegdist import vegdist

def test_vegdist_bray():
    # 2 samples, 3 species
    data = np.array([
        [10, 10, 10],
        [30, 0,  0]
    ])
    # manually calculate Bray-Curtis
    # sum of diffs: |10-30| + |10-0| + |10-0| = 20 + 10 + 10 = 40
    # sum of totals: 30 + 30 = 60
    # bray = 40 / 60 = 0.66666...
    res = vegdist(data, method="bray", upper=True)
    assert res.shape == (2, 2)
    assert np.isclose(res[0, 1], 2.0/3.0)

def test_vegdist_jaccard_binary():
    data = np.array([
        [10, 0, 10],
        [30, 1,  0]
    ])
    # binary: 
    # [1, 0, 1]
    # [1, 1, 0]
    # Intersection = 1 (feature 1)
    # Union = 3 (feature 1, 2, 3)
    # Jaccard dist = 1 - (1/3) = 2/3
    res = vegdist(data, method="jaccard", binary=True, upper=False)
    assert len(res) == 1 # Condensed 2x2 matrix has length 1
    assert np.isclose(res[0], 2.0/3.0)

def test_vegdist_jaccard_quantitative():
    data = np.array([
        [10, 10, 10],
        [30, 0,  0]
    ])
    # bray = 2/3
    # jaccard = 2 * (2/3) / (1 + 2/3) = (4/3) / (5/3) = 4/5 = 0.8
    res = vegdist(data, method="jaccard", binary=False)
    assert np.isclose(res[0], 0.8)
