# mapu

A native Python implementation of the classic R package [`vegan`](https://cran.r-project.org/web/packages/vegan/index.html) for community ecology.

This package aims to provide ecologists and data scientists using Python with rapid, pure-Python ports of popular `vegan` functions, including diversity indices, distance matrices (`vegdist`), ordination techniques, and multivariate statistical tests.

## Installation

You can install `mapu` using `pip`:

```bash
pip install mapu
```

Or using `uv` for ultra-fast dependency resolution and installation:

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
uv pip install -e ".[dev]"
```

## Features

The library currently supports a wide array of functions mirroring R `vegan`:

- **Diversity Analysis**: `diversity` (Shannon, Simpson, etc.), `fisher_alpha`, `specnumber`, `renyi`, `tsallis`, rarefaction functions (`rarefy`, `rrarefy`), and species accumulation curves.
- **Distance Matrices**: `vegdist` (Bray-Curtis, Jaccard, Euclidean, Manhattan, Gower, etc.), `designdist`.
- **Ordination**: Unconstrained (PCoA/`cmdscale`, PCA, NMDS/`metaMDS`, `isomap`) and Constrained (`rda`, `cca`, `capscale`).
- **Data Transformation**: `decostand`, `wisconsin`.
- **Statistical Tests**: `adonis` (PERMANOVA), `anosim`, `mantel`, `mrpp`, `simper`, `betadisper`.
- **Clustering**: `spantree`, `cascadeKM`, `cophenetic`.

## Example Usage

```python
import numpy as np
import pandas as pd
from mapu import vegdist, metaMDS, diversity

# Create a mock species abundance matrix (sites x species)
np.random.seed(42)
community_data = pd.DataFrame(np.random.poisson(lam=2, size=(10, 20)))

# Calculate Bray-Curtis distance
bray_dist = vegdist(community_data, method="bray")

# Perform Non-metric Multidimensional Scaling (NMDS)
nmds_result = metaMDS(community_data, distance="bray", k=2)
print("Stress:", nmds_result.stress)

# Calculate Shannon diversity
shannon_h = diversity(community_data, index="shannon")
print("Shannon Diversity:\\n", shannon_h)
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
