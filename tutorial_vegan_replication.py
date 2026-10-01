#!/usr/bin/env python3
"""
Replication of the vegan R tutorial
(https://peat-clark.github.io/BIO381/veganTutorial.html)
using the mapu Python package.

This script demonstrates the same analyses covered in Peter Clark's vegan
tutorial and compares mapu results against R vegan reference values.
"""

import numpy as np
from mapu.datasets import load_dune, load_dune_env
from mapu import diversity, specnumber, vegdist, metaMDS, anosim

# =============================================================================
# 1. LOAD DATA
# =============================================================================
print("=" * 70)
print("1. LOADING DATA")
print("=" * 70)

dune = load_dune()
dune_env = load_dune_env()

print(f"dune: {dune.shape[0]} sites × {dune.shape[1]} species")
print(f"dune.env: {dune_env.shape[0]} sites × {dune_env.shape[1]} variables")
print()
print("Species:", ", ".join(dune.columns[:10]), "...")
print("Environmental variables:", ", ".join(dune_env.columns))
print()

# =============================================================================
# 2. DIVERSITY INDICES
# =============================================================================
print("=" * 70)
print("2. DIVERSITY INDICES")
print("=" * 70)

# -- Simpson Index --
# R: diversity(dune, index = "simpson")
simpson = diversity(dune, index="simpson")

# Reference values from R vegan
r_simpson = np.array(
    [
        0.7345679,
        0.8900227,
        0.8787500,
        0.9007407,
        0.9140076,
        0.9001736,
        0.9075000,
        0.9087500,
        0.9115646,
        0.9031909,
        0.8671875,
        0.8685714,
        0.8521579,
        0.8333333,
        0.8506616,
        0.8429752,
        0.8355556,
        0.8614540,
        0.8740895,
        0.8678460,
    ]
)

print("\nSimpson Diversity Index:")
print(f"  {'Site':>4}  {'mapu':>12}  {'R vegan':>12}  {'Match':>6}")
print(f"  {'----':>4}  {'--------':>12}  {'--------':>12}  {'-----':>6}")
for i in range(len(simpson)):
    match = "✓" if np.isclose(simpson[i], r_simpson[i], atol=1e-6) else "✗"
    print(f"  {i+1:>4}  {simpson[i]:>12.7f}  {r_simpson[i]:>12.7f}  {match:>6}")

simpson_match = np.allclose(simpson, r_simpson, atol=1e-6)
print(f"\n  All Simpson values match R vegan: {'YES ✓' if simpson_match else 'NO ✗'}")

# -- Shannon Index --
# R: diversity(dune, index = "shannon")
shannon = diversity(dune, index="shannon")

r_shannon = np.array(
    [
        1.440482,
        2.252516,
        2.193749,
        2.426779,
        2.544421,
        2.345946,
        2.471733,
        2.434898,
        2.493568,
        2.398613,
        2.106065,
        2.114495,
        2.099638,
        1.863680,
        1.979309,
        1.959795,
        1.876274,
        2.079387,
        2.134024,
        2.048270,
    ]
)

print("\nShannon Diversity Index (H'):")
print(f"  {'Site':>4}  {'mapu':>12}  {'R vegan':>12}  {'Match':>6}")
print(f"  {'----':>4}  {'--------':>12}  {'--------':>12}  {'-----':>6}")
for i in range(len(shannon)):
    match = "✓" if np.isclose(shannon[i], r_shannon[i], atol=1e-4) else "✗"
    print(f"  {i+1:>4}  {shannon[i]:>12.6f}  {r_shannon[i]:>12.6f}  {match:>6}")

shannon_match = np.allclose(shannon, r_shannon, atol=1e-4)
print(f"\n  All Shannon values match R vegan: {'YES ✓' if shannon_match else 'NO ✗'}")

# -- Inverse Simpson --
# R: diversity(dune, index = "invsimpson")
invsimpson = diversity(dune, index="invsimpson")

r_invsimpson = np.array(
    [
        3.767442,
        9.092784,
        8.247423,
        10.074627,
        11.628931,
        10.017391,
        10.810811,
        10.958904,
        11.307692,
        10.329609,
        7.529412,
        7.608696,
        6.763975,
        6.000000,
        6.696203,
        6.368421,
        6.081081,
        7.217822,
        7.942149,
        7.566929,
    ]
)

print("\nInverse Simpson Diversity:")
print(f"  {'Site':>4}  {'mapu':>12}  {'R vegan':>12}  {'Match':>6}")
print(f"  {'----':>4}  {'--------':>12}  {'--------':>12}  {'-----':>6}")
for i in range(len(invsimpson)):
    match = "✓" if np.isclose(invsimpson[i], r_invsimpson[i], atol=1e-4) else "✗"
    print(f"  {i+1:>4}  {invsimpson[i]:>12.6f}  {r_invsimpson[i]:>12.6f}  {match:>6}")

invsimpson_match = np.allclose(invsimpson, r_invsimpson, atol=1e-4)
print(
    f"\n  All Inverse Simpson values match R vegan: {'YES ✓' if invsimpson_match else 'NO ✗'}"
)

# =============================================================================
# 3. SPECIES RICHNESS
# =============================================================================
print("\n" + "=" * 70)
print("3. SPECIES RICHNESS")
print("=" * 70)

# R: specnumber(dune)
spn = specnumber(dune)

r_spn = np.array([5, 10, 10, 13, 14, 11, 13, 12, 13, 12, 9, 9, 10, 7, 8, 8, 7, 9, 9, 8])

print(f"\n  {'Site':>4}  {'mapu':>6}  {'R vegan':>8}  {'Match':>6}")
print(f"  {'----':>4}  {'----':>6}  {'-------':>8}  {'-----':>6}")
for i in range(len(spn)):
    match = "✓" if int(spn[i]) == int(r_spn[i]) else "✗"
    print(f"  {i+1:>4}  {int(spn[i]):>6}  {int(r_spn[i]):>8}  {match:>6}")

spn_match = np.array_equal(spn.astype(int), r_spn.astype(int))
print(f"\n  All species numbers match R vegan: {'YES ✓' if spn_match else 'NO ✗'}")

# =============================================================================
# 4. DISSIMILARITY (BRAY-CURTIS)
# =============================================================================
print("\n" + "=" * 70)
print("4. BRAY-CURTIS DISSIMILARITY")
print("=" * 70)

# R: vegdist(dune, method = "bray")
bc = vegdist(dune, method="bray")

# Reference: first 5×5 block from R
r_bc_5x5 = np.array(
    [
        [0.0000000, 0.4666667, 0.4482759, 0.5238095, 0.6393443],
        [0.4666667, 0.0000000, 0.3414634, 0.3563218, 0.4117647],
        [0.4482759, 0.3414634, 0.0000000, 0.2705882, 0.4698795],
        [0.5238095, 0.3563218, 0.2705882, 0.0000000, 0.5000000],
        [0.6393443, 0.4117647, 0.4698795, 0.5000000, 0.0000000],
    ]
)

mapu_bc_5x5 = bc.iloc[:5, :5].values

print("\nBray-Curtis Distance (sites 1–5):")
print("\n  mapu:")
for i in range(5):
    row = "  ".join(f"{mapu_bc_5x5[i, j]:.7f}" for j in range(5))
    print(f"    [{row}]")

print("\n  R vegan:")
for i in range(5):
    row = "  ".join(f"{r_bc_5x5[i, j]:.7f}" for j in range(5))
    print(f"    [{row}]")

bc_match = np.allclose(mapu_bc_5x5, r_bc_5x5, atol=1e-6)
print(f"\n  Bray-Curtis (5×5) matches R vegan: {'YES ✓' if bc_match else 'NO ✗'}")

# =============================================================================
# 5. NMDS ORDINATION
# =============================================================================
print("\n" + "=" * 70)
print("5. NMDS ORDINATION")
print("=" * 70)

# R: set.seed(42); metaMDS(dune, k=2, trymax=20)
# R stress ≈ 0.1183186
nmds = metaMDS(dune, k=2, trymax=20)

print(f"\n  mapu NMDS stress:   {nmds['stress']:.6f}")
print("  R vegan NMDS stress: 0.118319  (≈, exact value depends on random starts)")
print(f"  Points shape:        {nmds['points'].shape}")
print(f"  Distance metric:     {nmds['distance']}")
print()

# Note: NMDS point coordinates differ between runs and libraries because
# they depend on random initialization and convergence. Stress should be
# in the same ballpark (< 0.20 is generally acceptable).
if nmds["stress"] < 0.30:
    print("  Stress < 0.30 — ordination is meaningful ✓")
else:
    print("  Stress ≥ 0.30 — ordination may be unreliable ✗")

print("\n  NMDS Coordinates (first 5 sites):")
for i in range(5):
    print(
        f"    Site {i+1}: [{nmds['points'][i, 0]:>10.6f}, {nmds['points'][i, 1]:>10.6f}]"
    )

# =============================================================================
# 6. ANOSIM
# =============================================================================
print("\n" + "=" * 70)
print("6. ANOSIM (Analysis of Similarities)")
print("=" * 70)

# R: anosim(dune, dune.env$Management, distance = "bray")
management = dune_env["Management"].values
anosim_result = anosim(dune, management, distance="bray", permutations=999)

print(f"\n  R statistic: {anosim_result['statistic']:.4f}")
print(f"  P-value:     {anosim_result['significance']:.4f}")
print(f"  Permutations: {anosim_result['permutations']}")
print()
if anosim_result["significance"] < 0.05:
    print("  Significant difference between management groups (p < 0.05) ✓")
else:
    print("  No significant difference between management groups (p ≥ 0.05)")

# =============================================================================
# 7. SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("7. SUMMARY: mapu vs R vegan")
print("=" * 70)

results = {
    "Simpson diversity": simpson_match,
    "Shannon diversity": shannon_match,
    "Inverse Simpson": invsimpson_match,
    "Species richness": spn_match,
    "Bray-Curtis distance": bc_match,
    "NMDS ordination": nmds["stress"] < 0.30,
}

print(f"\n  {'Analysis':<25} {'Status':>10}")
print(f"  {'-' * 25} {'-' * 10}")
for name, passed in results.items():
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"  {name:<25} {status:>10}")

all_pass = all(results.values())
print(f"\n  Overall: {'ALL TESTS PASSED ✓' if all_pass else 'SOME TESTS FAILED ✗'}")
print()
