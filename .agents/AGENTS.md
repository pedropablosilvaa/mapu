# mapu — Agent Instructions

## Overview

`mapu` is a native Python port of the [R vegan package](https://cran.r-project.org/package=vegan) for **community ecology** analysis. It provides functions for diversity analysis, ordination, dissimilarity computation, and hypothesis testing, with an API that mirrors R vegan as closely as possible.

## Project Structure

```
src/mapu/
├── __init__.py       # Public API exports
├── datasets.py       # Built-in datasets (dune, dune.env)
├── diversity.py      # Diversity indices (Shannon, Simpson, etc.)
├── vegdist.py        # Dissimilarity/distance matrices
├── ordination.py     # Ordination methods (PCoA, NMDS, CCA, RDA)
├── stats.py          # Statistical tests (ANOSIM, MANTEL, ADONIS, etc.)
├── transform.py      # Data transformations (decostand, wisconsin)
├── cluster.py        # Cluster analysis (spantree, cascadeKM)
└── viz.py            # Visualization utilities
```

## Key Design Decisions

- **Return types**: Functions return `np.ndarray`, `pd.DataFrame`, or `dict` depending on the context.
  - `vegdist()` returns a `pd.DataFrame` when input is a DataFrame, otherwise a condensed 1D array.
  - `metaMDS()` returns a dict with keys: `"points"`, `"stress"`, `"distance"`, `"k"`.
  - Statistical tests (e.g., `anosim`) return dicts with `"statistic"`, `"significance"`, `"permutations"`.
- **Data verification**: The `dune` dataset in `datasets.py` is verified against R vegan output (all 20 sites × 30 species). Any changes to dataset values must be cross-checked against R using `Rscript`.
- **R API compatibility**: Function names and parameter names follow R vegan conventions where possible (e.g., `trymax` as alias for `n_init` in `metaMDS`).

## Development Guidelines

1. **Always verify against R**: When adding or modifying ecological functions, validate results against R vegan using `Rscript` (R is available at `/usr/local/bin/R`).
2. **Handle DataFrame input**: Functions receiving data from `vegdist()` should handle `pd.DataFrame` input (which is a square distance matrix), converting to condensed form if needed via `squareform()`.
3. **Testing**: Run `tutorial_vegan_replication.py` after any changes to verify all analyses match R vegan output.
4. **Dependencies**: Core deps are `numpy`, `pandas`, `scipy`, `scikit-learn`. Use `uv` for package management.

## Datasets

- `load_dune()`: 20 sites × 30 species vegetation data from Dutch dune meadows.
- `load_dune_env()`: Environmental variables (A1, Moisture, Management, Use, Manure) for the 20 dune sites.

## Running the Tutorial Comparison

```bash
cd /Users/psilva/Documents/projects/ecology/package/new_package
.venv/bin/python tutorial_vegan_replication.py
```

This validates Simpson, Shannon, Inverse Simpson, species richness, Bray-Curtis, NMDS, and ANOSIM against R vegan reference values.
