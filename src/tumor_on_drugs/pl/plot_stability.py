"""Plots for stored stabilized-ICA stability results."""

from pathlib import Path

import numpy as np


def _get_mstd_stability(adata, key_added):
    entry = adata.uns.get(key_added)
    mstd = entry.get("mstd") if isinstance(entry, dict) else None
    if not isinstance(mstd, dict):
        raise KeyError(f"No MSTD results are stored in adata.uns[{key_added!r}].")
    component_counts = np.asarray(mstd["component_counts"], dtype=int)
    stability = np.asarray(mstd["stability_indexes"], dtype=float)
    if (
        component_counts.ndim != 1
        or component_counts.size == 0
        or np.any(component_counts < 1)
        or stability.ndim != 2
        or stability.shape[0] != len(component_counts)
        or np.any(component_counts > stability.shape[1])
    ):
        raise ValueError("Stored MSTD stability results have an invalid shape.")
    return component_counts, stability, mstd


def _resolve_optimal_n_compt(optimal_n_compt, mstd, component_counts):
    if optimal_n_compt is None:
        optimal_n_compt = mstd.get(
            "optimal_n_compt", mstd.get("selected_n_components")
        )
    if optimal_n_compt is None:
        return None
    if (
        isinstance(optimal_n_compt, (bool, np.bool_))
        or not isinstance(optimal_n_compt, (int, np.integer))
        or not 1 <= optimal_n_compt <= np.max(component_counts)
    ):
        raise ValueError(
            "optimal_n_compt must be a positive integer no greater than the "
            "largest tested component count, or None."
        )
    return int(optimal_n_compt)


def _finish_plot(fig, axis, *, save, return_fig, filename):
    import matplotlib.pyplot as plt

    if save is not None and save is not False:
        import scanpy as sc

        if save is True:
            save_path = Path(sc.settings.figdir) / f"{filename}.png"
        else:
            save_suffix = str(save)
            suffix_path = Path(save_suffix)
            if suffix_path.is_absolute():
                save_path = suffix_path
            else:
                if suffix_path.parent != Path("."):
                    save_path = Path(sc.settings.figdir) / suffix_path
                else:
                    stem = filename
                    if suffix_path.suffix:
                        suffix = save_suffix
                        if not suffix.startswith(("_", ".")):
                            suffix = f"_{suffix}"
                        save_path = Path(sc.settings.figdir) / f"{stem}{suffix}"
                    else:
                        save_path = (
                            Path(sc.settings.figdir)
                            / f"{stem}_{save_suffix}.png"
                        )
        if not save_path.suffix:
            save_path = save_path.with_suffix(".png")
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")
    if return_fig:
        return fig, axis
    plt.show()
    plt.close(fig)
    return None


def plot_all_stability_indices(
    adata,
    optimal_n_compt="elbow",
    *,
    key_added="ica",
    figsize=(10, 6),
    return_fig=False,
    save=None,
):
    """Plot MSTD stability curves and optionally mark the selected component count.

    ``optimal_n_compt="elbow"`` selects the elbow automatically.
    ``optimal_n_compt=None`` omits the vertical marker.
    """
    import matplotlib.pyplot as plt

    component_counts, stability, mstd = _get_mstd_stability(adata, key_added)
    if optimal_n_compt == "elbow":
        from ..pp.mstd import select_mstd_components

        optimal_n_compt = select_mstd_components(adata, key_added=key_added)
    elif optimal_n_compt is not None:
        optimal_n_compt = _resolve_optimal_n_compt(
            optimal_n_compt, mstd, component_counts
        )
    fig, axis = plt.subplots(figsize=figsize)
    for row, count in enumerate(component_counts):
        values = stability[row, :count]
        finite = np.isfinite(values)
        axis.plot(
            np.arange(1, count + 1)[finite],
            values[finite],
            label=f"{count} components",
        )
    if optimal_n_compt is not None:
        axis.axvline(
            optimal_n_compt,
            color="black",
            linestyle="--",
            label=f"Chosen number of components: {optimal_n_compt}",
        )
    axis.set_xlabel("Component Index")
    axis.set_ylabel("Stability Index")
    axis.set_ylim(0, 1)
    axis.set_title("Stability Indexes Across Component Counts")
    # axis.legend()
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _finish_plot(
        fig,
        axis,
        save=save,
        return_fig=return_fig,
        filename="plot_all_stability_indices",
    )


def plot_mean_stability_index(
    adata,
    *,
    key_added="ica",
    figsize=(10, 5),
    return_fig=False,
    save=None,
):
    """Plot MSTD stability distributions, optionally marking the selected elbow."""
    import matplotlib.pyplot as plt

    component_counts, stability, mstd = _get_mstd_stability(adata, key_added)
    distributions = [
        stability[row, :count][np.isfinite(stability[row, :count])]
        for row, count in enumerate(component_counts)
    ]
    fig, axis = plt.subplots(figsize=figsize)
    axis.boxplot(distributions, positions=component_counts, patch_artist=True)
    axis.set_xticks(component_counts)
    axis.set_xticklabels(component_counts)
    axis.set_xlabel("Number of components")
    axis.set_ylabel("Stability index")
    axis.set_ylim(0, 1)
    axis.set_title("Stabilized ICA component stability")
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _finish_plot(
        fig,
        axis,
        save=save,
        return_fig=return_fig,
        filename="plot_mean_stability_index",
    )
