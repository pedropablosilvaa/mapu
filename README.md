# mapu

[![CI](https://github.com/pedropablosilvaa/mapu/actions/workflows/ci.yml/badge.svg)](https://github.com/pedropablosilvaa/mapu/actions/workflows/ci.yml)
[![Docs](https://github.com/pedropablosilvaa/mapu/actions/workflows/docs.yml/badge.svg)](https://pedropablosilvaa.github.io/mapu/)
[![Python Versions](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/pedropablosilvaa/mapu)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A native Python implementation of the classic R package [`vegan`](https://cran.r-project.org/web/packages/vegan/index.html) for community ecology.

`mapu` provides ecologists, conservation biologists, and data scientists with pure-Python ports of popular `vegan` workflows—including diversity indices, dissimilarity matrices (`vegdist`), ordination methods (PCoA, NMDS, RDA, CCA), multivariate hypothesis testing (ANOSIM, ADONIS/PERMANOVA, Mantel), and built-in ecological datasets (`dune`, `dune.env`).

📖 **Full Documentation & API Reference**: [https://pedropablosilvaa.github.io/mapu/](https://pedropablosilvaa.github.io/mapu/)

---

## Installation

Install `mapu` using `pip`:

```bash
pip install mapu
```

Or using [`uv`](https://github.com/astral-sh/uv) for ultra-fast dependency resolution and installation:

```bash
uv pip install mapu
```

### Development Installation

To install from source for development using `uv`:

```bash
git clone https://github.com/pedropablosilvaa/mapu.git
cd mapu
uv venv
source .venv/bin/activate  # On Unix/macOS
uv pip install -e ".[dev,docs]"
```

---

## Features

- **Built-in Datasets**: `load_dune()`, `load_dune_env()` (Dutch Dune Meadow vegetation data verified directly against R `vegan`).
- **Diversity Analysis**: `diversity` (Shannon, Simpson, Inverse Simpson), `specnumber` (richness), `fisher_alpha`, `renyi`, `tsallis`, rarefaction (`rarefy`, `rrarefy`, `drarefy`), and species accumulation curves (`specaccum`, `poolaccum`).
- **Distance & Dissimilarity**: `vegdist` (Bray-Curtis, Jaccard, Euclidean, Manhattan, Gower, Kulczynski, Canberra, Horn, Mountford, Raup-Crick, Chord), `designdist`, `stepacross`.
- **Ordination**:
  - *Unconstrained*: NMDS (`metaMDS`), PCoA (`cmdscale`), PCA (`pca`), Isomap (`isomap`).
  - *Constrained*: RDA (`rda`), CCA (`cca`), db-RDA (`capscale`), Procrustes analysis (`procrustes`), environmental fitting (`envfit`).
- **Statistical Hypothesis Testing**: `anosim` (ANOSIM), `adonis` (PERMANOVA), `mantel` & `mantel_correlog`, `mrpp`, `simper`, `betadisper`, `bioenv`, `indval`.
- **Data Transformation**: `decostand` (total, max, freq, normalize, standardize, pa, chi.square, hellinger, log), `wisconsin`.
- **Clustering & Topology**: `spantree` (minimum spanning tree), `cascadeKM` (K-means cascade), `cophenetic`.
- **Visualization**: `ordiplot`, `ordihull`, `ordiellipse`, `ordispider`, `plot_specaccum`, `plot_rad`.

---

## Tutorial: Replicating R `vegan` with `mapu`

This tutorial replicates the standard community ecology analysis from Peter Clark's [vegan tutorial](https://peat-clark.github.io/BIO381/veganTutorial.html) using the classic Dutch Dune Meadow vegetation dataset (`dune`).

### 1. Load Data

```python
from mapu import (
    load_dune,
    load_dune_env,
    diversity,
    specnumber,
    vegdist,
    metaMDS,
    anosim,
)

# Load the classic 20 sites × 30 species dune vegetation data
dune = load_dune()
dune_env = load_dune_env()

print(f"Dune: {dune.shape[0]} sites × {dune.shape[1]} species")
print(f"Environmental variables: {list(dune_env.columns)}")
```

Output:
```text
Dune: 20 sites × 30 species
Environmental variables: ['A1', 'Moisture', 'Management', 'Use', 'Manure']
```

### 2. Alpha Diversity Indices

Alpha diversity measures the diversity of species within individual sites or samples:

```python
# Simpson's Diversity Index (1 - sum(p^2))
# Equivalent to R: diversity(dune, index = "simpson")
simpson_div = diversity(dune, index="simpson")
print("Simpson diversity (first 5 sites):", simpson_div[:5].round(4))

# Shannon Diversity Index (H' = -sum(p * ln(p)))
# Equivalent to R: diversity(dune, index = "shannon")
shannon_div = diversity(dune, index="shannon")
print("Shannon diversity (first 5 sites):", shannon_div[:5].round(4))

# Inverse Simpson Index (1 / sum(p^2))
# Equivalent to R: diversity(dune, index = "invsimpson")
inv_simpson = diversity(dune, index="invsimpson")
print("Inverse Simpson (first 5 sites):  ", inv_simpson[:5].round(4))

# Species Richness (count of unique observed species per site)
# Equivalent to R: specnumber(dune)
richness = specnumber(dune)
print("Species richness (first 5 sites): ", richness[:5])
```

Output:
```text
Simpson diversity (first 5 sites): [0.7346 0.89   0.8788 0.9007 0.914 ]
Shannon diversity (first 5 sites): [1.4405 2.2525 2.1937 2.4268 2.5444]
Inverse Simpson (first 5 sites):   [ 3.7674  9.0928  8.2474 10.0746 11.6289]
Species richness (first 5 sites):  [ 5 10 10 13 14]
```
*(All 20 sites produce identical numerical results to R `vegan`)*

### 3. Dissimilarity & Distance Matrices

Calculate pairwise Bray-Curtis dissimilarities between sites:

```python
# Equivalent to R: vegdist(dune, method = "bray")
bray_dist = vegdist(dune, method="bray")

# Display the first 5×5 distance matrix block
print(bray_dist.iloc[:5, :5].round(4))
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

### 4. Non-metric Multidimensional Scaling (NMDS)

Ordination in reduced 2D space preserving rank dissimilarity relationships:

```python
# Equivalent to R: set.seed(42); metaMDS(dune, k=2, trymax=20)
nmds = metaMDS(dune, distance="bray", k=2, trymax=20)

print(f"NMDS Stress: {nmds['stress']:.4f}")
print("Embedding coordinates shape:", nmds["points"].shape)
```

Output:
```text
NMDS Stress: 0.1183
Embedding coordinates shape: (20, 2)
```

### 5. Hypothesis Testing: ANOSIM

Test for significant differences in community composition across meadow management regimes (`Management`):

```python
# Equivalent to R: anosim(dune, dune.env$Management, distance = "bray")
management = dune_env["Management"].values
anosim_result = anosim(dune, grouping=management, distance="bray", permutations=999)

print(f"ANOSIM Statistic R:    {anosim_result['statistic']:.4f}")
print(f"Significance (p-value): {anosim_result['significance']:.4f}")
```

Output:
```text
ANOSIM Statistic R:    0.2579
Significance (p-value): 0.0060
```

With $p = 0.006 < 0.05$, meadow management significantly explains community composition differences between sites.

---

## Validation Summary: `mapu` vs R `vegan`

| Analysis | `mapu` | R `vegan` | Status |
| :--- | :--- | :--- | :--- |
| **Simpson Diversity** | Exact matches (20/20 sites) | Reference values | ✓ PASS |
| **Shannon Diversity** | Exact matches ($<10^{-6}$ diff) | Reference values | ✓ PASS |
| **Inverse Simpson** | Exact matches ($<10^{-6}$ diff) | Reference values | ✓ PASS |
| **Species Richness** | Exact integer matches (20/20 sites) | Reference values | ✓ PASS |
| **Bray-Curtis Matrix**| Exact matches (all pairwise distances) | Reference values | ✓ PASS |
| **NMDS Stress** | $\approx 0.12$ | $\approx 0.118$ | ✓ PASS |
| **ANOSIM Statistic $R$** | $0.2579$ | $0.2579$ | ✓ PASS |

To run the automated validation suite comparing all analyses against R reference values locally:

```bash
python tutorial_vegan_replication.py
```

---

## Documentation

- **Documentation Home**: [https://pedropablosilvaa.github.io/mapu/](https://pedropablosilvaa.github.io/mapu/)
- **Tutorial Guide**: [https://pedropablosilvaa.github.io/mapu/tutorial/](https://pedropablosilvaa.github.io/mapu/tutorial/)
- **API Reference**: [https://pedropablosilvaa.github.io/mapu/api/ordination/](https://pedropablosilvaa.github.io/mapu/api/ordination/)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
