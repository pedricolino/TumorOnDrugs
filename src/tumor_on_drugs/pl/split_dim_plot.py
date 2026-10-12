import math
import os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap, Normalize
from matplotlib import cm


def split_dim_plot(
    adata,
    split_by: str,
    basis: str = "umap",
    color: Optional[str] = None,
    ncols: int = 5,
    cmap: str = None,
    figsize: Optional[tuple] = None,
    s: int = 8,
    alpha: float = 1.0,
    show_background: bool = True,
    bg_color: str = "lightgrey",
    bg_alpha: float = 0.25,
    bg_s: Optional[int] = None,
    return_fig: bool = False,
    save: Optional[str] = None,
    rasterized: bool = False,
    figdir: Optional[str] = None,
):
    """
    Plot basis split by a categorical column (split_by). Fixes categorical handling,
    ensures continuous color scales are equal across subplots, and optionally draws
    faint background cells in grey for context.

    save behavior:
    - save=None: do not save
    - save=True: save with a default filename
    - save=str: save using the provided suffix/name, Scanpy-style
    """

    def _default_filename():
        parts = ["split_dim_plot", basis, split_by]
        if color is not None:
            parts.append(color)
        return "_".join(parts) + ".png"

    def _resolve_save_path(save_value):
        if save_value is None or save_value is False:
            return None

        # Determine filename
        if save_value is True:
            filename = _default_filename()
        else:
            save_str = str(save_value)

            # Scanpy-style: if user passes "foo.pdf" or "_foo.pdf"
            if save_str.startswith("."):
                # treat as suffix-like string
                filename = _default_filename().replace(".png", save_str)
            elif os.path.splitext(save_str)[1]:
                # explicit extension/path given
                filename = save_str
            else:
                # suffix without extension
                filename = _default_filename().replace(".png", f"_{save_str}.png")

        # If figdir is given and filename is not absolute, place it there
        path = Path(filename)
        if not path.is_absolute() and figdir is not None:
            path = Path(figdir) / path

        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    # --- Get basis matrix ---
    emb_key_candidates = [f"X_{basis}", basis]
    emb = None
    for k in emb_key_candidates:
        if k in getattr(adata, "obsm", {}):
            emb = adata.obsm[k]
            break
    if emb is None:
        raise KeyError(
            f"Couldn't find basis in adata.obsm. Tried keys: {emb_key_candidates}"
        )
    if emb.shape[1] < 2:
        raise ValueError(f"basis '{basis}' does not appear to be 2D.")

    X = emb[:, 0]
    Y = emb[:, 1]

    # --- Validate split_by ---
    if split_by not in adata.obs.columns:
        raise KeyError(f"split_by='{split_by}' not found in adata.obs columns.")

    groups = pd.Categorical(adata.obs[split_by]).categories
    if len(groups) == 0:
        groups = pd.unique(adata.obs[split_by].fillna("").values)

    n_panels = len(groups)
    nrow = math.ceil(n_panels / ncols)
    if figsize is None:
        figsize = (4 * ncols, 4 * nrow)

    fig, axes = plt.subplots(nrow, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    # --- Prepare color handling ---
    is_categorical = False
    category_colors = None
    norm = None
    cmap_obj = cm.get_cmap(cmap) if cmap is not None else None

    if color is not None:
        if color not in adata.obs.columns:
            raise KeyError(f"color='{color}' not found in adata.obs columns.")

        col_series = adata.obs[color]

        if pd.api.types.is_categorical_dtype(col_series.dtype) or pd.api.types.is_object_dtype(
            col_series.dtype
        ):
            is_categorical = True
            adata.obs[color] = col_series.astype("category")
            col_series = adata.obs[color]
            categories = list(col_series.cat.categories)

            provided_colors = adata.uns.get(color + "_colors", None)
            if provided_colors is not None and len(provided_colors) >= len(categories):
                category_colors = provided_colors[: len(categories)]
            else:
                base = cm.get_cmap("tab20")
                category_colors = [base(i / max(1, len(categories) - 1)) for i in range(len(categories))]

            discrete_cmap = ListedColormap(category_colors)
            cat_norm = mpl.colors.BoundaryNorm(np.arange(-0.5, len(categories) + 0.5), len(categories))
        else:
            numeric = pd.to_numeric(col_series, errors="coerce")
            if numeric.isna().all():
                raise ValueError(f"color column '{color}' could not be converted to numeric values.")
            vmin = numeric.min()
            vmax = numeric.max()
            if np.isclose(vmin, vmax):
                delta = 0.5 if vmin == 0 else abs(0.1 * vmin)
                vmin -= delta
                vmax += delta
            norm = Normalize(vmin=vmin, vmax=vmax)

    if bg_s is None:
        bg_s = max(1, int(s * 0.8))

    # --- Plot each group ---
    for ax_idx, grp in enumerate(groups):
        ax = axes_flat[ax_idx]
        mask = adata.obs[split_by] == grp
        xi = X[mask.values]
        yi = Y[mask.values]

        if show_background:
            ax.scatter(
                X, Y,
                s=bg_s,
                color=bg_color,
                alpha=bg_alpha,
                linewidths=0,
                zorder=0,
                rasterized=rasterized,
            )

        if color is None:
            ax.scatter(
                xi, yi,
                s=s,
                alpha=alpha,
                linewidths=0,
                color="k",
                zorder=1,
                rasterized=rasterized,
            )
            ax.set_title(str(grp))
        else:
            col_vals = adata.obs.loc[mask, color]
            if is_categorical:
                codes = col_vals.cat.codes.values
                ax.scatter(
                    xi, yi,
                    c=codes,
                    cmap=discrete_cmap,
                    norm=cat_norm,
                    s=s,
                    alpha=alpha,
                    linewidths=0,
                    zorder=1,
                    rasterized=rasterized,
                )
                ax.set_title(str(grp))
            else:
                numeric_vals = pd.to_numeric(col_vals, errors="coerce").values
                ax.scatter(
                    xi, yi,
                    c=numeric_vals,
                    cmap=cmap_obj,
                    norm=norm,
                    s=s,
                    alpha=alpha,
                    linewidths=0,
                    zorder=1,
                    rasterized=rasterized,
                )
                ax.set_title(str(grp))

        ax.set_xticks([])
        ax.set_yticks([])

    for empty_ax in axes_flat[n_panels:]:
        empty_ax.axis("off")

    # --- Add shared legend / colorbar ---
    if color is not None:
        if is_categorical:
            from matplotlib.patches import Patch
            handles = [Patch(color=category_colors[i], label=str(cat)) for i, cat in enumerate(categories)]
            fig.legend(handles=handles, loc="center right", title=color)
            fig.subplots_adjust(right=0.85)
        else:
            cax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
            sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
            sm.set_array([])
            fig.colorbar(sm, cax=cax, label=color)

    plt.tight_layout(rect=[0, 0, 0.9, 1.0] if (color is not None) else None)

    # --- Save like Scanpy ---
    save_path = _resolve_save_path(save)
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    if return_fig:
        return fig, axes

    plt.show()
    plt.close(fig)