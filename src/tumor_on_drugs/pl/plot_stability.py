"""Plots for stored stabilized-ICA stability results."""

import numpy as np


def plot_mean_stability_index(adata, *, key_added="ica", figsize=(10, 5)):
    """Plot per-component stability distributions from ``adata.uns`` MSTD results."""
    import matplotlib.pyplot as plt

    entry = adata.uns.get(key_added)
    mstd = entry.get("mstd") if isinstance(entry, dict) else None
    if not isinstance(mstd, dict):
        raise KeyError(f"No MSTD results are stored in adata.uns[{key_added!r}].")
    component_counts = np.asarray(mstd["component_counts"], dtype=int)
    stability = np.asarray(mstd["stability_indexes"], dtype=float)
    if stability.ndim != 2 or stability.shape[0] != len(component_counts):
        raise ValueError("Stored MSTD stability results have an invalid shape.")

    distributions = [
        stability[row, :count][np.isfinite(stability[row, :count])]
        for row, count in enumerate(component_counts)
    ]
    fig, axis = plt.subplots(figsize=figsize)
    axis.boxplot(distributions, patch_artist=True)
    axis.set_xticks(np.arange(1, len(component_counts) + 1))
    axis.set_xticklabels(component_counts)
    axis.set_xlabel("Number of components")
    axis.set_ylabel("Stability index")
    axis.set_title("Stabilized ICA component stability")
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig, axis
