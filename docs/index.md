# mapu

[![CI](https://github.com/pedropablosilvaa/mapu/actions/workflows/ci.yml/badge.svg)](https://github.com/pedropablosilvaa/mapu/actions/workflows/ci.yml)
[![Docs](https://github.com/pedropablosilvaa/mapu/actions/workflows/docs.yml/badge.svg)](https://pedropablosilvaa.github.io/mapu/)
[![Python Versions](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)](https://github.com/pedropablosilvaa/mapu)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A native Python implementation of the classic R package [`vegan`](https://cran.r-project.org/web/packages/vegan/index.html) for community ecology.

This package aims to provide ecologists and data scientists using Python with rapid, pure-Python ports of popular `vegan` functions, including diversity indices, distance matrices (`vegdist`), ordination techniques, and multivariate statistical tests.

---

## Installation

You can install `mapu` using `pip`:

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

The library currently supports a wide array of functions mirroring R `vegan`:

- **Built-in Datasets**: `load_dune()`, `load_dune_env()`.
- **Diversity Analysis**: `diversity` (Shannon, Simpson, Inverse Simpson), `specnumber`, `fisher_alpha`, `renyi`, `tsallis`, rarefaction functions (`rarefy`, `rrarefy`, `drarefy`), and species accumulation curves (`specaccum`, `poolaccum`).
- **Distance Matrices**: `vegdist` (Bray-Curtis, Jaccard, Euclidean, Manhattan, Gower, etc.), `designdist`, `stepacross`.
- **Ordination**: Unconstrained (PCoA/`cmdscale`, PCA, NMDS/`metaMDS`, `isomap`) and Constrained (`rda`, `cca`, `capscale`).
- **Data Transformation**: `decostand`, `wisconsin`.
- **Statistical Tests**: `adonis` (PERMANOVA), `anosim`, `mantel`, `mrpp`, `simper`, `betadisper`.
- **Clustering**: `spantree`, `cascadeKM`, `cophenetic`.
- **Visualization**: `ordiplot`, `ordihull`, `ordiellipse`, `ordispider`, `plot_specaccum`, `plot_rad`.

---

## Quick Tutorial: Replicating R `vegan`

Below is an overview of replicating the classic R `vegan` workflow using the built-in Dutch Dune Meadow dataset. For the full in-depth walkthrough, see the dedicated [Tutorial](tutorial.md).

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

# 1. Load dune vegetation and environmental datasets
dune = load_dune()
dune_env = load_dune_env()

# 2. Compute diversity indices (matching R vegan)
simpson_div = diversity(dune, index="simpson")
shannon_div = diversity(dune, index="shannon")
richness = specnumber(dune)

# 3. Compute Bray-Curtis distance matrix
bray_dist = vegdist(dune, method="bray")

# 4. Perform NMDS ordination
nmds = metaMDS(dune, distance="bray", k=2, trymax=20)
print("NMDS Stress:", nmds["stress"])

# 5. Test management effects with ANOSIM
management = dune_env["Management"].values
anosim_result = anosim(dune, grouping=management, distance="bray", permutations=999)
print(f"ANOSIM R: {anosim_result['statistic']:.4f}, p-value: {anosim_result['significance']:.4f}")
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
