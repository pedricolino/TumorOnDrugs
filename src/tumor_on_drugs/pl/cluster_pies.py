import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Wedge
import scanpy as sc


def sc_pl_cluster_pies(
    adata,
    cluster_key="leiden",
    obs_key="phase",
    basis="umap",                    # scanpy style: "umap" -> uses adata.obsm["X_umap"]
    ax=None,
    figsize=(7, 7),
    underlay=True,                   # draw scanpy embedding first
    underlay_color=None,             # default: cluster_key
    underlay_kwargs=None,            # passed to sc.pl.embedding
    radius_base=0.035,               # relative to embedding span
    radius_scale_by_cluster_size=True,
    min_fraction_to_draw=0.0,
    palette="tab20",                 # fallback if adata.uns[f"{obs_key}_colors"] absent
    pie_alpha=0.9,                   # 90% opacity default
    edgecolor="white",
    linewidth=0.4,
    show=True,
    save=False,                      # False | True | "filename.png"
    dpi=120,
    return_data=True,
    verbose=True,
):
    """
    Draw cluster-level pies for obs_key composition, optionally on top of sc.pl.embedding.

    Save semantics (scanpy-like):
      - save=False: don't save
      - save=True: auto file name
      - save='name.png': explicit file name
    """
    if cluster_key not in adata.obs:
        raise ValueError(f"{cluster_key} not found in adata.obs")
    if obs_key not in adata.obs:
        raise ValueError(f"{obs_key} not found in adata.obs")
    basis_key = basis if basis.startswith("X_") else f"X_{basis}"
    if basis_key not in adata.obsm:
        raise ValueError(f"{basis_key} not found in adata.obsm")
    if not (0.0 <= pie_alpha <= 1.0):
        raise ValueError("pie_alpha must be between 0 and 1.")

    cl = adata.obs[cluster_key].astype(str)
    cat = adata.obs[obs_key].astype(str)
    coords = np.asarray(adata.obsm[basis_key])[:, :2].astype(float)

    # cluster centroids + composition table
    dfc = pd.DataFrame({"cluster": cl.values, "x": coords[:, 0], "y": coords[:, 1]})
    centroids = dfc.groupby("cluster", sort=False)[["x", "y"]].mean()
    cluster_sizes = dfc.groupby("cluster", sort=False).size()

    counts = pd.crosstab(cl, cat, dropna=False)
    props = counts.div(counts.sum(axis=1), axis=0).fillna(0.0)

    clusters = props.index.tolist()
    categories = props.columns.tolist()

    # pie colors: use adata.uns[f"{obs_key}_colors"] if present
    obs_uns_key = f"{obs_key}_colors"
    pie_colors = {}
    used_obs_uns = False
    if obs_uns_key in adata.uns:
        uns_colors = list(adata.uns[obs_uns_key])
        if len(uns_colors) >= len(categories):
            pie_colors = {c: uns_colors[i] for i, c in enumerate(categories)}
            used_obs_uns = True

    if not used_obs_uns:
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

    # create axis if needed
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        created_fig = True
    else:
        fig = ax.figure

    # underlay with scanpy embedding
    if underlay:
        if underlay_kwargs is None:
            underlay_kwargs = {}
        color_key = underlay_color if underlay_color is not None else cluster_key
        sc.pl.embedding(
            adata,
            basis=basis,
            color=color_key,
            ax=ax,
            show=False,
            **underlay_kwargs,
        )

    # radius in embedding coordinate system
    xspan = max(coords[:, 0].max() - coords[:, 0].min(), 1e-12)
    yspan = max(coords[:, 1].max() - coords[:, 1].min(), 1e-12)
    span = max(xspan, yspan)
    base_r = radius_base * span
    sref = np.sqrt(cluster_sizes.max()) if radius_scale_by_cluster_size else 1.0

    # overlay pies
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
                    (cx, cy),
                    r,
                    start,
                    end,
                    facecolor=pie_colors[cat_name],
                    edgecolor=edgecolor,
                    linewidth=linewidth,
                    alpha=pie_alpha,
                    zorder=20,
                )
            )
            start = end

    # keep scanpy look
    ax.set_aspect("equal", adjustable="datalim")

    # save behavior
    saved_path = None
    if save is not False:
        if save is True:
            saved_path = f"{obs_key}_per_{cluster_key}_{basis}.png"
        elif isinstance(save, str):
            saved_path = save
        else:
            raise ValueError("save must be False, True, or a filename string.")
        fig.savefig(saved_path, bbox_inches="tight", dpi=dpi)
        if verbose:
            print(f"Saved: {saved_path}")

    if show and created_fig:
        plt.show()
    elif created_fig and not show:
        plt.close(fig)

    out = {
        "ax": ax,
        "saved_path": saved_path,
        "n_clusters": int(len(clusters)),
        "categories": categories,
        "pie_colors": pie_colors,
    }
    if return_data:
        out["proportions"] = props
        out["counts"] = counts
    return out