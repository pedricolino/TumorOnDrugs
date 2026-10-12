import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.patches import Wedge, Patch


def add_pie_plot_overlay(
    adata,
    ax=None,
    cluster_key="leiden",
    obs_key="phase",
    basis="umap",                    # uses adata.obsm["X_umap"] if "umap"
    radius_base=0.035,               # relative to embedding span
    radius_scale_by_cluster_size=True,
    min_fraction_to_draw=0.0,
    pie_alpha=0.9,                   # 90% opacity default
    edgecolor="white",
    linewidth=0.4,
    palette="tab20",                 # fallback if adata.uns[f"{obs_key}_colors"] absent
    add_legend=True,
    legend_title=None,
    legend_loc="upper left",
    legend_bbox_to_anchor=(1.02, 1.0),
    legend_frameon=False,
    legend_kwargs=None,
):
    """
    Overlay cluster-level pie charts (obs_key composition) on an existing embedding axis.

    Expected usage:
        sc.pl.embedding(..., ax=ax, show=False)
        add_cluster_pies_overlay(adata, ax=ax, ...)

    Returns dict with proportions/counts/colors.
    """
    if cluster_key not in adata.obs:
        raise ValueError(f"{cluster_key} not found in adata.obs")
    if obs_key not in adata.obs:
        raise ValueError(f"{obs_key} not found in adata.obs")
    if ax is None:
        import matplotlib.pyplot as plt
        ax = plt.gca()
    if not (0.0 <= pie_alpha <= 1.0):
        raise ValueError("pie_alpha must be between 0 and 1.")

    basis_key = basis if basis.startswith("X_") else f"X_{basis}"
    if basis_key not in adata.obsm:
        raise ValueError(f"{basis_key} not found in adata.obsm")

    cl = adata.obs[cluster_key].astype(str)
    cat = adata.obs[obs_key].astype(str)
    coords = np.asarray(adata.obsm[basis_key])[:, :2].astype(float)

    # centroids and proportions
    dfc = pd.DataFrame({"cluster": cl.values, "x": coords[:, 0], "y": coords[:, 1]})
    centroids = dfc.groupby("cluster", sort=False)[["x", "y"]].mean()
    cluster_sizes = dfc.groupby("cluster", sort=False).size()

    counts = pd.crosstab(cl, cat, dropna=False)
    props = counts.div(counts.sum(axis=1), axis=0).fillna(0.0)

    clusters = props.index.tolist()
    categories = props.columns.tolist()

    # category colors (for pies + legend)
    uns_key = f"{obs_key}_colors"
    pie_colors = {}
    used_uns = False
    if uns_key in adata.uns:
        uns_colors = list(adata.uns[uns_key])
        if len(uns_colors) >= len(categories):
            pie_colors = {c: uns_colors[i] for i, c in enumerate(categories)}
            used_uns = True

    if not used_uns:
        if isinstance(palette, str):
            cmap = mpl.colormaps[palette]
            pie_colors = {
                c: mpl.colors.to_hex(cmap(i / max(len(categories), 1)))
                for i, c in enumerate(categories)
            }
        else:
            if len(palette) < len(categories):
                raise ValueError("Provided palette list shorter than number of categories.")
            pie_colors = {c: palette[i] for i, c in enumerate(categories)}

    # radius in embedding coordinate system
    xspan = max(coords[:, 0].max() - coords[:, 0].min(), 1e-12)
    yspan = max(coords[:, 1].max() - coords[:, 1].min(), 1e-12)
    span = max(xspan, yspan)
    base_r = radius_base * span
    sref = np.sqrt(cluster_sizes.max()) if radius_scale_by_cluster_size else 1.0

    # draw pies
    for cl_name in clusters:
        cx, cy = centroids.loc[cl_name, "x"], centroids.loc[cl_name, "y"]

        if radius_scale_by_cluster_size:
            r = base_r * (np.sqrt(cluster_sizes.loc[cl_name]) / max(sref, 1e-12))
            r = max(r, base_r * 0.35)
        else:
            r = base_r

        row = props.loc[cl_name].copy()
        if min_fraction_to_draw > 0:
            row[row < min_fraction_to_draw] = 0.0
            if row.sum() > 0:
                row = row / row.sum()

        start = 0.0
        for cat_name, frac in row.items():
            if frac <= 0:
                continue
            end = start + float(frac) * 360.0
            ax.add_patch(
                Wedge(
                    (cx, cy), r, start, end,
                    facecolor=pie_colors[cat_name],
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                    alpha=pie_alpha,
                    zorder=30,
                )
            )
            start = end

    # legend for pie categories
    if add_legend:
        handles = [Patch(facecolor=pie_colors[c], edgecolor="none", label=str(c), alpha=pie_alpha) for c in categories]
        if legend_kwargs is None:
            legend_kwargs = {}
        ax.legend(
            handles=handles,
            title=(obs_key if legend_title is None else legend_title),
            loc=legend_loc,
            bbox_to_anchor=legend_bbox_to_anchor,
            frameon=legend_frameon,
            **legend_kwargs,
        )

    return {
        "n_clusters": int(len(clusters)),
        "categories": categories,
        "pie_colors": pie_colors,
        "proportions": props,
        "counts": counts,
    }