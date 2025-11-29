"""
Multi-dimensional Scaling (MDS) Implementation

This script illustrates the difference between Metric (Classical) and
Non-Metric MDS (NMDS) on a dataset of 20 randomly generated 2D points,
where noise has been deliberately introduced into the pairwise distance matrix.

The final result aligns both reconstructed solutions (MDS and NMDS)
with the ground truth data using PCA for visual comparison.
"""

# Authors: Wallace de H. Costa

# --- 1. Library Imports ---
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
from sklearn.metrics import euclidean_distances

# --- 2. Data Generation and Preparation ---
print("Generating and preparing data points...")

# Configuration constants
N_POINTS = 20
RANDOM_SEED = 3
JITTER_MAGNITUDE = 0.5  # Controls the amount of noise added to distances
MIN_FLOAT_EPS = np.finfo(np.float32).eps

# Initialize Random State for reproducibility
rng = np.random.RandomState(seed=RANDOM_SEED)

# Generate 2D coordinates (Ground Truth)
# Uniformly generate points in a 20x20 area
X_ground_truth = rng.randint(0, 20, 2 * N_POINTS).astype(float).reshape((N_POINTS, 2))

# Center the data around the origin
X_ground_truth -= X_ground_truth.mean(axis=0)

# Compute pairwise Euclidean distances
D_true = euclidean_distances(X_ground_truth)

# Introduce symmetric noise (jitter) to the distance matrix
D_noise = rng.rand(N_POINTS, N_POINTS) * JITTER_MAGNITUDE
D_noise = D_noise + D_noise.T  # Ensure symmetry
np.fill_diagonal(D_noise, 0)
D_input = D_true + D_noise

# --- 3. Multi-dimensional Scaling Reconstruction ---
print("Performing Metric and Non-Metric MDS reconstructions...")
N_COMPONENTS = 2

# 3.1. Metric MDS (Classical MDS)
# Dissimilarities are treated as true distances.
mds_metric = MDS(
    n_components=N_COMPONENTS,
    max_iter=3000,
    eps=1e-9,
    n_init=1,
    random_state=42,
    dissimilarity="precomputed",
    n_jobs=1,
)
X_metric_reco = mds_metric.fit_transform(D_input)

# 3.2. Non-Metric MDS (NMDS)
# Dissimilarities are treated only as rankings (ordinal data).
mds_non_metric = MDS(
    n_components=N_COMPONENTS,
    metric=False,  # Key setting for Non-Metric MDS
    max_iter=3000,
    eps=1e-12,
    dissimilarity="precomputed",
    random_state=42,
    n_jobs=1,
    n_init=1,
)
X_non_metric_reco = mds_non_metric.fit_transform(D_input)


# --- 4. Post-processing and Alignment ---

# Rescale NMDS solution to match the variance/spread of the true data
norm_factor = np.sqrt((X_ground_truth**2).sum()) / np.sqrt((X_non_metric_reco**2).sum())
X_non_metric_reco *= norm_factor

# 4.1. Rotation and Centering using PCA
# PCA is used to align the orientation of the three point clouds
pca_aligner = PCA(n_components=N_COMPONENTS)
X_true_pca = pca_aligner.fit_transform(X_ground_truth)
X_metric_pca = pca_aligner.fit_transform(X_metric_reco)
X_non_metric_pca = pca_aligner.fit_transform(X_non_metric_reco)

# 4.2. Flip Axes to Match Orientation
# Ensure the sign of the PCA components is aligned for visual consistency
for i in range(N_COMPONENTS):
    # Align Metric MDS solution
    if np.corrcoef(X_metric_pca[:, i], X_true_pca[:, i])[0, 1] < 0:
        X_metric_pca[:, i] *= -1
    # Align Non-Metric MDS solution
    if np.corrcoef(X_non_metric_pca[:, i], X_true_pca[:, i])[0, 1] < 0:
        X_non_metric_pca[:, i] *= -1

# --- 5. Visualization ---
print("Generating plot...")

fig, ax = plt.subplots(figsize=(8, 8))
POINT_SIZE = 100

# 5.1. Scatter Plots
ax.scatter(X_true_pca[:, 0], X_true_pca[:, 1], color="darkblue", s=POINT_SIZE, alpha=1.0, label="Ground Truth")
ax.scatter(X_metric_pca[:, 0], X_metric_pca[:, 1], color="cyan", s=POINT_SIZE, alpha=0.9, label="Metric MDS")
ax.scatter(X_non_metric_pca[:, 0], X_non_metric_pca[:, 1], color="orangered", s=POINT_SIZE, alpha=0.9, label="Non-Metric MDS")

# 5.2. Plotting Edges (True Distance Visualization)
# Lines are drawn between true points, with thickness related to inverse distance.
segments = [
    [X_true_pca[i, :], X_true_pca[j, :]]
    for i in range(N_POINTS)
    for j in range(N_POINTS)
]

# Edge weight calculation: higher value means a shorter distance (stronger link)
edge_weights = D_input.max() / (D_input + MIN_FLOAT_EPS) * 100
np.fill_diagonal(edge_weights, 0)
edge_weights = np.abs(edge_weights)

# Create Line Collection
line_coll = LineCollection(
    segments,
    zorder=0,
    cmap=plt.cm.Blues,
    norm=plt.Normalize(0, edge_weights.max())
)
line_coll.set_array(edge_weights.flatten())
line_coll.set_linewidths(np.full(len(segments), 0.5))

# Add lines to the plot
ax.add_collection(line_coll)

# Final plot configuration
ax.legend(scatterpoints=1, loc="best", frameon=True, title="MDS Solutions")
ax.set_title(f"Comparison of Metric and Non-Metric MDS (N={N_POINTS})")
plt.axis('equal') # Ensure aspect ratio is preserved
plt.tight_layout()
plt.show()
