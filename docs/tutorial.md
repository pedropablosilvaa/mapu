# Community Ecology Tutorial: Replicating R `vegan` with `mapu`

This tutorial walks through a standard community ecology workflow using `mapu`, replicating the analyses from Peter Clark's well-known [vegan tutorial](https://peat-clark.github.io/BIO381/veganTutorial.html) using the classic Dutch Dune Meadow vegetation dataset (`dune`).

All calculations in `mapu` produce exact or numerical matches to R's `vegan` package.

---

## 1. Setup and Loading Data

`mapu` provides built-in loaders for the standard Dutch dune meadow dataset:
- `load_dune()`: 20 sites × 30 species abundance matrix.
- `load_dune_env()`: 20 sites × 5 environmental variables (`A1`, `Moisture`, `Management`, `Use`, `Manure`).

```python
import numpy as np
import pandas as pd
from mapu import (
    load_dune,
    load_dune_env,
    diversity,
    specnumber,
    vegdist,
    metaMDS,
    anosim,
)

# Load data
dune = load_dune()
dune_env = load_dune_env()

print(f"Dune vegetation: {dune.shape[0]} sites × {dune.shape[1]} species")
print(f"Environmental variables: {list(dune_env.columns)}")
```

Output:
```text
Dune vegetation: 20 sites × 30 species
Environmental variables: ['A1', 'Moisture', 'Management', 'Use', 'Manure']
```

---

## 2. Alpha Diversity Indices

Alpha diversity measures the diversity of species within individual sites or samples.

### Simpson's Diversity Index
Simpson's index ($D_1 = 1 - \sum p_i^2$) quantifies the probability that two individuals randomly selected from a sample belong to different species.

```python
# Compute Simpson's index
simpson_div = diversity(dune, index="simpson")
print("Simpson diversity (first 5 sites):")
print(simpson_div[:5])
```

Output:
```text
[0.7345679  0.89002268 0.87875    0.90074074 0.91400762]
```
*(Identical to R `vegan::diversity(dune, index = "simpson")`)*

### Shannon Diversity Index
Shannon's entropy ($H' = -\sum p_i \ln p_i$) measures uncertainty and accounts for both abundance and evenness.

```python
# Compute Shannon's index
shannon_div = diversity(dune, index="shannon")
print("Shannon diversity (first 5 sites):")
print(shannon_div[:5])
```

Output:
```text
[1.44048158 2.25251641 2.19374944 2.42677894 2.54442111]
```
*(Identical to R `vegan::diversity(dune, index = "shannon")`)*

### Inverse Simpson Index
The inverse Simpson index ($1 / \sum p_i^2$) expresses diversity in terms of effective number of species.

```python
inv_simpson = diversity(dune, index="invsimpson")
print("Inverse Simpson diversity (first 5 sites):")
print(inv_simpson[:5])
```

Output:
```text
[ 3.76744186  9.09278351  8.24742268 10.07462687 11.62893082]
```
*(Identical to R `vegan::diversity(dune, index = "invsimpson")`)*

### Species Richness
Species richness ($S$) is the count of distinct species observed per site.

```python
richness = specnumber(dune)
print("Species richness (first 5 sites):")
print(richness[:5])
```

Output:
```text
[ 5 10 10 13 14]
```
*(Identical to R `vegan::specnumber(dune)`)*

---

## 3. Dissimilarity and Distance Matrices

`vegdist` computes pairwise ecological dissimilarity matrices between sites. The **Bray-Curtis** index is the standard metric for species abundance counts:

$$d_{jk} = \frac{\sum_i |x_{ij} - x_{ik}|}{\sum_i (x_{ij} + x_{ik})}$$

```python
# Compute Bray-Curtis distance matrix
bray_curtis = vegdist(dune, method="bray")

# View pairwise distance for first 5 sites
print(bray_curtis.iloc[:5, :5].round(4))
```

Output:
```text
        1       2       3       4       5
1  0.0000  0.4667  0.4483  0.5238  0.6393
2  0.4667  0.0000  0.3415  0.3563  0.4118
3  0.4483  0.3415  0.0000  0.2706  0.4699
4  0.5238  0.3563  0.2706  0.0000  0.5000
5  0.6393  0.4118  0.4699  0.5000  0.0000
```
*(Matches R `vegan::vegdist(dune, method = "bray")`)*

---

## 4. Non-metric Multidimensional Scaling (NMDS)

NMDS uses iterative rank-based optimization to place sites in low-dimensional space (typically 2D) preserving relative ecological dissimilarities.

`metaMDS()` mimics `vegan::metaMDS`:

```python
# Perform NMDS ordination (2 dimensions, Bray-Curtis distance)
nmds = metaMDS(dune, distance="bray", k=2, trymax=20)

print(f"NMDS Stress: {nmds['stress']:.4f}")
print("Site coordinates shape:", nmds["points"].shape)
```

Output:
```text
NMDS Stress: 0.1183  # (approximate, depending on random start)
Site coordinates shape: (20, 2)
```

> **Rule of Thumb for Stress**:
> - $< 0.05$: Excellent representation
> - $< 0.10$: Good representation with little danger of false inference
> - $< 0.20$: Usable representation; fine details should be treated cautiously
> - $> 0.20$: May lead to misleading interpretations

---

## 5. Hypothesis Testing: ANOSIM

Analysis of Similarities (**ANOSIM**) tests whether there are statistically significant differences in species composition among pre-defined groups (e.g. `Management` type: BF, HF, NM, SF).

The test statistic $R$ ranges from $-1$ to $+1$:
- $R = 1$: Complete separation between groups.
- $R = 0$: No difference between groups (random composition).

```python
# Test differences among Management types
management = dune_env["Management"].values
anosim_result = anosim(dune, grouping=management, distance="bray", permutations=999)

print(f"ANOSIM Statistic R: {anosim_result['statistic']:.4f}")
print(f"Significance (p-value): {anosim_result['significance']:.4f}")
```

Output:
```text
ANOSIM Statistic R: 0.2579
Significance (p-value): 0.0060
```

With $p = 0.006 < 0.05$, we conclude that meadow management significantly influences vegetation community composition.

---

## 6. Summary of Validation Against R `vegan`

| Metric / Analysis | `mapu` | R `vegan` | Status |
| :--- | :--- | :--- | :--- |
| **Simpson Diversity** | Exact matches (all 20 sites) | Reference values | ✓ PASS |
| **Shannon Diversity** | Exact matches ($<10^{-6}$ diff) | Reference values | ✓ PASS |
| **Inverse Simpson** | Exact matches ($<10^{-6}$ diff) | Reference values | ✓ PASS |
| **Species Richness** | Exact integer matches | Reference values | ✓ PASS |
| **Bray-Curtis Matrix**| Exact matches (all pairs) | Reference values | ✓ PASS |
| **NMDS Stress** | $\approx 0.12$ | $\approx 0.118$ | ✓ PASS |
| **ANOSIM $R$** | $0.2579$ | $0.2579$ | ✓ PASS |
