"""
rdx.py - RDX section 3 + visualizations implementation adapted from https://github.com/nkondapa/RDX

Paper: https://arxiv.org/abs/2505.23917  (Kondapaneni, Mac Aodha, Perona, 2025)

>>> # D_a, D_b: symmetric (N, N) pairwise distance matrices (need not be Euclidean)
>>> R_a = distance_to_rank(D_a)
>>> R_b = distance_to_rank(D_b)
>>> G_ab = compute_G(R_a, R_b, gamma=0.4)   # difference matrix, A-->B direction
>>> F_ab = compute_F(G_ab, beta=5)          # affinity matrix (symmetrised)
>>> labels = cluster_F(F_ab, n_clusters=8)
>>> Z_a, Z_b = dim_project(D_a), pca_project(D_b)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import Literal
from sklearn.cluster import SpectralClustering
from sklearn.manifold import MDS


# 3.1 - Distance --> Neighbourhood Rank matrix

def distance_to_rank(D: np.ndarray) -> np.ndarray:
    """Convert a pairwise distance matrix to a neighbourhood-rank matrix.

    For each row i, sample j receives the rank of d(i, j) among all distances
    from i (rank 0 = closest neighbour, rank N-1 = farthest).  The diagonal
    (self-distance) is naturally rank 0 for every row.

    Parameters
    ----------
    D : (N, N) symmetric distance matrix (numpy float array).

    Returns
    -------
    R : (N, N) float array of neighbourhood ranks.
    """
    D = np.asarray(D, dtype=np.float32)
    sort_order = np.argsort(D, axis=1)          # indices that sort each row
    R = np.argsort(sort_order, axis=1).astype(np.float32)  # rank of each entry
    return R

# Step 3.2 - Difference matrix G_{A,B}

def compute_G(
    R_a: np.ndarray,
    R_b: np.ndarray,
    gamma: float | None = None,
    gamma_scale: float | None = None,
) -> np.ndarray:
    """Compute the signed difference matrix G_{A,B}.

    G_{A,B}[i, j] encodes how much closer sample j is to i in representation A
    than in representation B, based on neighbourhood ranks.  Negative values
    mean items i and j are closer in A than in B..
    
    G[i,j] = tanh(\\gamma * (R_b[i,j] - R_a[i,j]) / (min(R_a, R_b)[i,j] + 1))

    Parameters
    ----------
    R_a, R_b    : (N, N) rank matrices from `distance_to_rank`.
    gamma       : Sensitivity parameter.  If None, set to gamma_scale / N.
    gamma_scale : Used only when gamma is None (default 10).

    Returns
    -------
    G : (N, N) float array; positive where A groups i,j closer than B does.
    """
    R_a = np.asarray(R_a, dtype=np.float32)
    R_b = np.asarray(R_b, dtype=np.float32)
    denom = np.minimum(R_a, R_b) + (1.0 if np.isclose(np.min(np.minimum(R_a, R_b)), 0) else 0)          # +1 avoids division by zero on the diagonal
    diff = R_b - R_a

    N = R_a.shape[0]
    if gamma is None:
        if gamma_scale is None:
            # So that G(mean_magnitude) = 0.5
            mean = np.mean(np.abs(diff) / denom)
            gamma = np.arctanh(0.5) / (mean+1e-12)
        else:
            gamma = gamma_scale / N
    return np.tanh(gamma * diff / denom)


# 3.3 - Affinity matrix F_{A,B}  ( matrix that gets clustered)

def compute_F(G: np.ndarray, beta: float = 5.0) -> np.ndarray:
    """Compute the RDX affinity matrix F_{A,B} from the difference matrix G_{A,B}.

    Directly from the paper (Section 3):
        F_{A,B} = exp(-\\beta * G_{A,B})

    The matrix is then symmetrised by averaging with its transpose:
        F_{A,B} = (F_{A,B} + F_{A,B}.T) / 2

    Parameters
    ----------
    G    : (N, N) difference matrix from `compute_G`.
    beta : Amplification factor
    Returns
    -------
    F : (N, N) symmetric affinity matrix ready for spectral clustering.
    """
    G = np.asarray(G, dtype=np.float32)
    F = np.exp(-beta * G)
    F = (F + F.T) / 2.0
    return F


# 3.4 - Spectral clustering of F_{A,B}

def cluster_F(
    F: np.ndarray,
    n_clusters: int = 10,
    random_state: int = 0,
) -> np.ndarray:
    """Cluster the RDX affinity matrix F_{A,B} using spectral clustering.

    After clustering the labels are re-ordered by increasing mean intra-cluster
    affinity, so that cluster 0 always has the lowest internal coherence (it
    acts as a null / background cluster) and higher-numbered clusters correspond
    to progressively tighter, more distinctive groups.

    Parameters
    ----------
    F            : (N, N) symmetric affinity matrix from `compute_F`.
    n_clusters   : Number of clusters (K in the paper).
    random_state : Seed for reproducibility.

    Returns
    -------
    labels : (N,) integer array of cluster assignments (0-indexed).
    """
    F = np.asarray(F, dtype=np.float64)

    sc = SpectralClustering(
        n_clusters=n_clusters,
        affinity="precomputed",
        random_state=random_state,
    )
    sc.fit(F)
    raw_labels = sc.labels_

    # Re-order clusters by mean intra-cluster affinity (ascending)
    unique_labels = np.unique(raw_labels)
    mean_aff = np.array([
        F[raw_labels == k][:, raw_labels == k].mean()
        for k in unique_labels
    ])
    rank_order = np.argsort(mean_aff) # lowest affinity --> label 0

    label_map = {old: new for new, old in enumerate(rank_order)}
    labels = np.array([label_map[l] for l in raw_labels], dtype=np.int32)
    return labels


# 2-D projection for visualisation (MDS, PCoA, or a shared dispatcher)

def pca_project(
    D: np.ndarray,
    n_components: int = 2,
) -> np.ndarray:
    """PCoA (principal coordinates analysis) from a distance matrix.

    Double-centers the squared distance matrix and takes the leading
    eigenvectors; this is the standard linear embedding used when a Euclidean
    configuration is implied (often called classical MDS or PCoA).

    Parameters
    ----------
    D            : (N, N) symmetric distance matrix (precomputed, any metric).
    n_components : Target dimensionality (default 2 for scatter plots).

    Returns
    -------
    Z : (N, n_components) projected coordinates.
    """
    D = np.asarray(D, dtype=np.float64)
    n = D.shape[0]
    D2 = D * D
    H = np.eye(n, dtype=np.float64) - np.ones((n, n), dtype=np.float64) / n
    B = -0.5 * (H @ D2 @ H)
    eigvals, eigvecs = np.linalg.eigh(B)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    Z = np.zeros((n, n_components), dtype=np.float64)
    c = 0
    for j in range(n):
        if c >= n_components:
            break
        lam = eigvals[j]
        if lam > 0:
            Z[:, c] = eigvecs[:, j] * np.sqrt(lam)
            c += 1
    return Z


def mds_project(
    D: np.ndarray,
    n_components: int = 2,
    n_init: int = 4,
    random_state: int = 0,
    **kwargs,
) -> np.ndarray:
    """Metric MDS (SMACOF) from a distance matrix.

    Parameters
    ----------
    D            : (N, N) symmetric distance matrix (precomputed, any metric).
    n_components : Target dimensionality (default 2 for scatter plots).
    n_init       : Number of restarts; the run with lowest stress is returned.
    random_state : Reproducibility seed.

    See sklearn.manifold.MDS

    Returns
    -------
    Z : (N, n_components) projected coordinates.
    """
    D = np.asarray(D, dtype=np.float64)
    mds = MDS(
        n_components=n_components,
        dissimilarity="precomputed",
        metric=True,
        normalized_stress=True,
        n_init=n_init,
        random_state=random_state,
        **kwargs
    )
    return mds.fit_transform(D)


def dim_project(
    D: np.ndarray,
    n_components: int = 2,
    *,
    projection_mode: Literal["mds", "pca"] = "mds",
    n_init: int = 4,
    random_state: int = 0,
    **mds_kwargs,
) -> np.ndarray:
    """Project from a distance matrix, dispatching to `mds_project` or `pca_project`.

    Parameters
    ----------
    D               : (N, N) symmetric distance matrix.
    n_components    : Output dimensionality.
    projection_mode : `"mds"` (SMACOF) or `"pca"` (PCoA / classical MDS).
    n_init, random_state, **mds_kwargs
        Forwarded to `mds_project` only when `projection_mode="mds"`.
    """
    D = np.asarray(D, dtype=np.float64)
    if projection_mode == "pca":
        return pca_project(D, n_components=n_components)
    if projection_mode == "mds":
        return mds_project(
            D,
            n_components=n_components,
            n_init=n_init,
            random_state=random_state,
            **mds_kwargs,
        )
    raise ValueError("projection_mode must be 'mds' or 'pca'")


# Convenience wrapper - run the full pipeline in one call

def rdx(
    D_a: np.ndarray,
    D_b: np.ndarray,
    *,
    beta: float = 5.0,
    gamma: float | None = None,
    gamma_scale: float = 80.0,
    n_clusters: int = 10,
    random_state: int = 0,
) -> dict:
    """Run the full RDX pipeline for both directions (A-->B and B-->A).

    Given distance matrices D_a and D_b this returns the M_A matrices and
    cluster labels for both the "A explains what A has over B" direction and
    the symmetric "B explains what B has over A" direction.

    For 2-D visualisation, pass `D_a` and `D_b` to `dim_project` or
    `mds_project` (SMACOF) / `pca_project` (PCoA) as needed.

    Parameters
    ----------
    D_a, D_b     : (N, N) pairwise distance matrices (need not be Euclidean).
    beta         : Affinity decay rate.
    gamma        : Difference sensitivity (None --> gamma_scale / N).
    gamma_scale  : Fallback scale for gamma.
    n_clusters   : Number of spectral clusters.
    random_state : Reproducibility seed.

    Returns
    -------
    dict with keys:

    `"R_a"`, `"R_b"`
        Neighbourhood rank matrices.
    `"G_ab"`
        Difference matrix G_{A,B} (positive = closer in A).
    `"G_ba"`
        Difference matrix G_{B,A} (positive = closer in B).
    `"F_ab"`
        RDX affinity matrix F_{A,B}, highlighting structure unique to A.
    `"F_ba"`
        RDX affinity matrix F_{B,A}, highlighting structure unique to B.
    `"labels_ab"`
        Cluster labels based on F_ab (A-specific clusters).
    `"labels_ba"`
        Cluster labels based on F_ba (B-specific clusters).
    """
    R_a = distance_to_rank(D_a)
    R_b = distance_to_rank(D_b)

    G_ab = compute_G(R_a, R_b, gamma=gamma, gamma_scale=gamma_scale)
    G_ba = compute_G(R_b, R_a, gamma=gamma, gamma_scale=gamma_scale)

    F_ab = compute_F(G_ab, beta=beta)
    F_ba = compute_F(G_ba, beta=beta)

    labels_ab = cluster_F(F_ab, n_clusters=n_clusters, random_state=random_state)
    labels_ba = cluster_F(F_ba, n_clusters=n_clusters, random_state=random_state)

    return dict(
        R_a=R_a, R_b=R_b,
        G_ab=G_ab, G_ba=G_ba,
        F_ab=F_ab, F_ba=F_ba,
        labels_ab=labels_ab, labels_ba=labels_ba,
    )


# Visualization helpers


def _axis_labels_for_projection(projection_mode: str) -> tuple[str, str]:
    if projection_mode == "pca":
        return "PC 1", "PC 2"
    return "MDS 1", "MDS 2"


def _cluster_cmap(n: int):
    """Return a ListedColormap with n visually distinct colours.

    Uses tab20 for n ≤ 20, then falls back to HSV for larger counts.
    Cluster 0 is always rendered in light gray (background / null cluster).
    """
    if n <= 20:
        base = plt.get_cmap("tab20")(np.linspace(0, 1, 20))
    else:
        base = plt.get_cmap("hsv")(np.linspace(0, 1, n, endpoint=False))
    # Override cluster-0 colour to light gray
    colors = base[:n].copy()
    colors[0] = [0.75, 0.75, 0.75, 0.35]
    return mcolors.ListedColormap(colors)


def plot_clusters(
    Z: np.ndarray,
    labels: np.ndarray,
    *,
    ax: Axes | None = None,
    title: str = "",
    point_size: float = 18.0,
    alpha: float = 0.85,
    show_legend: bool = True,
    projection_mode: Literal["mds", "pca"] = "mds",
):
    """Scatter plot of a 2-D embedding (MDS or PCoA) coloured by RDX cluster labels.

    Cluster 0 (the low-affinity background cluster) is plotted first in gray
    so that higher-affinity clusters sit on top visually.

    """
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    unique = np.unique(labels)
    cmap = _cluster_cmap(len(unique))

    for i, cl in enumerate(unique):
        mask = labels == cl
        color = cmap(i)
        a = 0.25 if cl == 0 else alpha
        ax.scatter(
            Z[mask, 0], Z[mask, 1],
            c=[color], s=point_size, alpha=a,
            linewidths=0, label=f"cluster {cl}",
            zorder=1 if cl == 0 else 2,
        )

    xlabel, ylabel = _axis_labels_for_projection(projection_mode)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=8)

    if show_legend and len(unique) <= 20:
        ax.legend(
            fontsize=7, markerscale=1.4,
            loc="best", framealpha=0.7,
            ncol=max(1, len(unique) // 10),
        )

    return ax


def plot_affinity_matrix(
    M: np.ndarray,
    labels: np.ndarray | None = None,
    *,
    ax: Axes | None = None,
    title: str = "",
    cmap: str = "viridis",
):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))

    M = np.asarray(M)
    if labels is not None:
        order = np.argsort(labels)
        M = M[order][:, order]

    im = ax.imshow(M, cmap=cmap, aspect="auto", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    if labels is not None:
        sorted_labels = np.sort(labels)
        boundaries = np.where(np.diff(sorted_labels))[0] + 0.5
        for b in boundaries:
            ax.axhline(b, color="magenta", linewidth=0.6, alpha=0.8)
            ax.axvline(b, color="magenta", linewidth=0.6, alpha=0.8)

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("sample index", fontsize=9)
    ax.set_ylabel("sample index", fontsize=9)
    ax.tick_params(labelsize=8)

    return ax


def plot_rdx_overview(
    D_a: np.ndarray,
    D_b: np.ndarray,
    result: dict,
    *,
    direction: str = "ab",
    name_a: str = "Model A",
    name_b: str = "Model B",
    figsize: tuple[float, float] = (16, 8),
    projection_mode: Literal["mds", "pca"] = "mds",
    mds_kwargs: dict | None = None,
) -> Figure:
    """Six-panel overview figure matching the layout in the RDX paper.
    """
    extra = {**(mds_kwargs or {})}
    extra.pop("projection_mode", None)
    labels = result[f"labels_{direction}"]
    F      = result[f"F_{direction}"]

    Z_a = dim_project(
        D_a, projection_mode=projection_mode, **extra,
    )
    Z_b = dim_project(
        D_b, projection_mode=projection_mode, **extra,
    )

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    fig.subplots_adjust(hspace=0.35, wspace=0.3)

    src, tgt = (name_a, name_b) if direction == "ab" else (name_b, name_a)
    
    plot_clusters(Z_a, labels, ax=axes[0, 0],
                  title=f"{name_a}  (clusters from {src}-->{tgt})",
                  projection_mode=projection_mode)
    plot_clusters(Z_b, labels, ax=axes[0, 1],
                  title=f"{name_b}  (same cluster colours)",
                  show_legend=False, projection_mode=projection_mode)

    # Points coloured by mean per-sample G value (how much A differs from B)
    mean_G = result[f"G_{direction}"].mean(axis=1)
    sc = axes[0, 2].scatter(
        Z_a[:, 0], Z_a[:, 1],
        c=mean_G, cmap="bwr",
        s=18, alpha=0.8, linewidths=0,
    )
    plt.colorbar(sc, ax=axes[0, 2], fraction=0.046, pad=0.04)
    axes[0, 2].set_title(f"Mean $G_{{A,B}}$ per sample  ({src}-->{tgt})", fontsize=10)
    x0, y0 = _axis_labels_for_projection(projection_mode)
    axes[0, 2].set_xlabel(x0, fontsize=9)
    axes[0, 2].set_ylabel(y0, fontsize=9)
    axes[0, 2].tick_params(labelsize=8)


    plot_affinity_matrix(
        D_a, labels, ax=axes[1, 0],
        title=f"{name_a} distance matrix", cmap="Blues",
    )
    plot_affinity_matrix(
        D_b, labels, ax=axes[1, 1],
        title=f"{name_b} distance matrix", cmap="Blues",
    )
    plot_affinity_matrix(
        F, labels, ax=axes[1, 2],
        title=f"RDX affinity $F_{{A,B}}$  ({src}-->{tgt})", cmap="viridis",
    )

    fig.suptitle(
        f"RDX overview — {src} vs {tgt}", fontsize=12, y=1.01,
    )
    return fig