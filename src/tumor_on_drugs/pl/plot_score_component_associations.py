"""Plot precomputed score-to-component associations from AnnData."""

import numpy as np
import pandas as pd


def plot_score_component_associations(
    adata,
    score_key=None,
    *,
    key_added="ica",
    component_label=None,
    correlation_threshold=0.3,
    mutual_info_threshold=0.3,
    figsize=None,
):
    """Plot whichever correlation and mutual-information metrics are stored.

    Associations must first be computed with
    :func:`tumor_on_drugs.pp.compute_score_component_associations`.
    """
    import matplotlib.pyplot as plt

    entry = adata.uns.get(key_added)
    associations = entry.get("associations") if isinstance(entry, dict) else None
    if not isinstance(associations, dict) or not associations:
        raise KeyError(
            f"No score associations found in adata.uns[{key_added!r}]; "
            "compute them with pp.compute_score_component_associations first."
        )
    if score_key is None:
        if len(associations) != 1:
            raise ValueError(
                "score_key must be specified when multiple score associations are stored."
            )
        score_key = next(iter(associations))
    if score_key not in associations:
        raise KeyError(f"No associations for score {score_key!r} are stored.")

    stored = associations[score_key]
    indices = np.asarray(stored["pc_index"], dtype=int)
    component_names = np.asarray(
        stored.get("component_names", [f"IC{i}" for i in indices]), dtype=str
    )
    if component_label is None:
        embedding_key = str(stored.get("embedding_key", "X_ica"))
        component_label = "PC" if "pca" in embedding_key.lower() else "IC"
    table = {
        "pc_index": indices,
        "component": component_names,
    }
    available_metrics = []
    if "mutual_info" in stored:
        table["mutual_info"] = np.asarray(stored["mutual_info"], dtype=float)
        available_metrics.append(("mutual_info", mutual_info_threshold, "Mutual information"))
    if "correlation" in stored:
        table["correlation"] = np.asarray(stored["correlation"], dtype=float)
        table["abs_correlation"] = np.asarray(
            stored.get("abs_correlation", np.abs(table["correlation"])), dtype=float
        )
        available_metrics.append(("abs_correlation", correlation_threshold, "|correlation|"))
    if not available_metrics:
        raise ValueError(
            f"Stored associations for {score_key!r} contain no plottable metrics."
        )
    if "p_value" in stored:
        table["p_value"] = np.asarray(stored["p_value"], dtype=float)
    frame = pd.DataFrame(table).sort_values("pc_index")

    if figsize is None:
        figsize = (6 * len(available_metrics), 4)
    fig, axes = plt.subplots(1, len(available_metrics), figsize=figsize, squeeze=False)
    axes = axes.ravel()
    for axis, (metric, threshold, label) in zip(axes, available_metrics):
        values = frame[metric].to_numpy(dtype=float)
        colors = np.full(len(values), "#2c7bb6", dtype=object)
        if "p_value" in frame:
            significant = (frame["p_value"].to_numpy(dtype=float) < 0.05) & (
                values >= threshold
            )
            colors[significant] = "#9c9696"
        axis.bar(
            np.arange(len(frame)),
            values,
            color=colors,
            edgecolor="white",
            linewidth=0.5,
        )
        axis.axhline(
            threshold,
            color="grey",
            linestyle="--",
            linewidth=1,
            label=f"{label} = {threshold:g}",
        )
        axis.set_ylabel(label)
        axis.set_title(f"{score_key} vs {component_label}s")
        axis.set_xticks(np.arange(len(frame)))
        axis.set_xticklabels(frame["component"], rotation=45, ha="right", fontsize=8)
        axis.legend(fontsize=8)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(False)
    fig.tight_layout()
    return fig, axes
