"""Plot treatment and timepoint trends for cluster proportions."""


def plot_cluster_trends(
    adata,
    cluster_col=None,
    treatment_col=None,
    timepoint_col=None,
    marker="",
    fig_size=None,
    save=None,
    return_fig=False,
    legend_placement=None,
    colors=None,
):
    import matplotlib.pyplot as plt
    import pandas as pd
    import scanpy as sc

    if cluster_col is None:
        raise ValueError("cluster_col must be specified.")

    treatment_col = treatment_col or (
        "treatment" if "treatment" in adata.obs.columns else None
    )
    timepoint_col = timepoint_col or (
        "week" if "week" in adata.obs.columns else None
    )

    if treatment_col is None or timepoint_col is None:
        raise ValueError(
            "treatment_col and timepoint_col must be specified when defaults "
            "are unavailable."
        )

    treatments = adata.obs[treatment_col].dropna().unique().tolist()
    if not treatments:
        raise ValueError("No non-missing treatments were found.")
    if fig_size is None:
        fig_size = (6 * len(treatments), 4)

    legend_i = len(treatments) - 1 if legend_placement is None else legend_placement
    cluster_values = adata.obs[cluster_col]
    if hasattr(cluster_values.dtype, "categories"):
        cluster_order = [str(x) for x in cluster_values.cat.categories]
    else:
        cluster_order = sorted(cluster_values.dropna().astype(str).unique())

    color_map = dict(zip(cluster_order, colors)) if colors is not None else {}
    fig, axes = plt.subplots(1, len(treatments), figsize=fig_size, squeeze=False)
    axes = axes.ravel()

    for index, treatment in enumerate(treatments):
        axis = axes[index]
        subset = adata.obs.loc[adata.obs[treatment_col] == treatment].copy()
        counts = (
            subset.groupby(timepoint_col, observed=False)[cluster_col]
            .value_counts(normalize=True)
            .unstack(fill_value=0)
        )
        counts.index = pd.to_numeric(counts.index)
        counts = counts.sort_index()
        counts.columns = counts.columns.astype(str)
        counts = counts.reindex(columns=cluster_order, fill_value=0)

        for cluster in counts.columns:
            axis.plot(
                counts.index,
                counts[cluster],
                marker=marker,
                label=f"Cluster {cluster}",
                color=color_map.get(cluster),
            )
        axis.set_title(str(treatment))
        axis.set_xlabel("Week")
        axis.set_ylabel("Fraction of cells")
        legend = axis.get_legend()
        if index == legend_i:
            axis.legend(
                title=cluster_col,
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
            )
        elif legend is not None:
            legend.remove()
    plt.tight_layout()
    if save is not None:
        plt.savefig(sc.settings.figdir / save, bbox_inches="tight")
    if return_fig:
        return fig, axes
    plt.show()
    plt.close(fig)
