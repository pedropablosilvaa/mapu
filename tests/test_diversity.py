import numpy as np
import pandas as pd
import pytest
from mapu.diversity import diversity, specnumber

def test_diversity_shannon():
    # Simple data matrix: 2 samples, 3 species
    # Sample 1: uniform distribution -> highest Shannon
    # Sample 2: diverse, but un-even
    data = np.array([
        [10, 10, 10],
        [30, 0,  0]
    ])
    
    # R vegan check:
    # > diversity(c(10,10,10))
    # [1] 1.098612
    # > diversity(c(30,0,0))
    # [1] 0
    res = diversity(data, index="shannon")
    
    assert len(res) == 2
    assert np.isclose(res[0], 1.098612, atol=1e-5)
    assert np.isclose(res[1], 0.0, atol=1e-5)

def test_diversity_simpson():
    data = np.array([
        [10, 10, 10],
        [30, 0,  0]
    ])
    # For [10,10,10], p = [0.333, 0.333, 0.333]
    # sum(p^2) = 0.333
    # simpson = 1 - 0.333 = 0.6666667
    res = diversity(data, index="simpson")
    assert np.isclose(res[0], 2.0/3.0)
    assert np.isclose(res[1], 0.0)

def test_diversity_invsimpson():
    data = np.array([
        [10, 10, 10],
        [30, 0,  0]
    ])
    # sum(p^2) = 0.333, invsimpson = 3
    # for [30,0,0], sum(p^2) = 1, invsimpson = 1
    res = diversity(data, index="invsimpson")
    assert np.isclose(res[0], 3.0)
    assert np.isclose(res[1], 1.0)

def test_specnumber():
    data = np.array([
        [10, 10, 10],
        [30, 0,  0]
    ])
    res = specnumber(data)
    assert np.array_equal(res, [3, 1])
