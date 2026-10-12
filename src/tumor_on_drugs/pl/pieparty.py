import time
import colorsys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Wedge
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import scipy.sparse as sp

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


def _log(msg, verbose=True):
    if verbose:
        print(msg, flush=True)


def _lighten_color(color, amount=1.0):
    r, g, b = mpl.colors.to_rgb(color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    rr, gg, bb = colorsys.hls_to_rgb(h, 1 - amount * (1 - l), s)
    return (rr, gg, bb)


def _resolve_color(base_color, percentile):
    if isinstance(base_color, str) and base_color.startswith("@"):
        return mpl.colors.to_rgb("#" + base_color[1:])
    cmap = mpl.colormaps[base_color] if base_color in mpl.colormaps else plt.get_cmap(base_color)
    return cmap(float(percentile))[:3]


def _get_gene_vector(X, gi):
    col = X[:, gi]
    if sp.issparse(col):
        return np.asarray(col.toarray()).ravel()
    return np.asarray(col).ravel()


def pieparty_from_adata(
    adata,
    gene_lists,
    colors,
    output_filename="pieparty.png",
    embedding_key="X_umap",
    expression_layer=None,
    proportionalize=True,
    cutoff_percentage=1.0,
    cutoff_expression_value=0.0,
    lighten_colors=True,
    lighten_colors_based_on_global=True,
    plot_resolution=3000,
    pie_sizes_scaling_factor=1.0,
    giant_pies=False,
    cluster_key=None,
    dpi=220,
    verbose=True,
    show_progress=True,
    random_state=0,
    max_cells=None,          # e.g. 20000 for speed
):
    t0 = time.time()
    rng = np.random.default_rng(random_state)

    def _iter(it, desc):
        if show_progress and tqdm is not None:
            return tqdm(it, desc=desc)
        return it

    _log("[1/8] Validating inputs...", verbose)
    if len(colors) != len(gene_lists):
        raise ValueError("Number of colors must equal number of gene lists.")
    if embedding_key not in adata.obsm:
        raise ValueError(f"{embedding_key} not found in adata.obsm")

    # optional subsample
    n_obs = adata.n_obs
    if max_cells is not None and n_obs > max_cells:
        idx_keep = rng.choice(n_obs, size=max_cells, replace=False)
        idx_keep.sort()
        _log(f"  Subsampling cells: {n_obs} -> {len(idx_keep)}", verbose)
        ad = adata[idx_keep].copy()
    else:
        ad = adata

    X = ad.layers[expression_layer] if expression_layer is not None else ad.X
    coords = np.asarray(ad.obsm[embedding_key])[:, :2].astype(float)
    genes = np.asarray(ad.var_names)
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    needed = sorted({g for gl in gene_lists for g in gl})
    missing = [g for g in needed if g not in gene_to_idx]
    if missing:
        raise ValueError(f"Missing genes in adata.var_names (first 20): {missing[:20]}")

    _log("[2/8] Preparing coordinates...", verbose)
    x = coords[:, 0]
    y = coords[:, 1]
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()
    xr = max(xmax - xmin, 1e-12)
    yr = max(ymax - ymin, 1e-12)

    W = int(plot_resolution * 1.1)
    H = int(plot_resolution * 1.1)
    px = (x - xmin) / xr * plot_resolution + 50
    py = (1 - (y - ymin) / yr) * plot_resolution + 50
    r_base = pie_sizes_scaling_factor * 30 * plot_resolution / 2400 / 2.0

    _log("[3/8] Loading expression vectors...", verbose)
    gene_vec = {}
    for g in _iter(needed, "Loading genes"):
        gene_vec[g] = _get_gene_vector(X, gene_to_idx[g]).astype(np.float32)

    _log("[4/8] Building per-cell matrix...", verbose)
    # Flatten genes in draw order
    gene_order = []
    color_bases = []
    group_sizes = []
    group_index = []
    for li, gl in enumerate(gene_lists):
        n = max(len(gl), 1)
        for gj, g in enumerate(gl):
            gene_order.append(g)
            color_bases.append(_resolve_color(colors[li], (gj / n) * 0.999))
            group_sizes.append(n)
            group_index.append(li)

    G = len(gene_order)
    N = ad.n_obs
    M = np.empty((N, G), dtype=np.float32)
    for j, g in enumerate(_iter(gene_order, "Stacking matrix")):
        M[:, j] = gene_vec[g]

    # giant pies aggregation
    idx_by_cl = None
    if giant_pies:
        _log("[5/8] Aggregating giant pies by cluster...", verbose)
        if cluster_key is None or cluster_key not in ad.obs:
            raise ValueError("cluster_key is required when giant_pies=True")
        cl = ad.obs[cluster_key].astype(str).values
        uniq = list(dict.fromkeys(cl))
        idx_by_cl = [np.where(cl == c)[0] for c in uniq]

        bg_px, bg_py = px.copy(), py.copy()  # background grey points
        px = np.array([px[idx].mean() for idx in idx_by_cl], dtype=np.float32)
        py = np.array([py[idx].mean() for idx in idx_by_cl], dtype=np.float32)

        # preserve original script behavior (/n cluster) after mean
        M2 = np.empty((len(uniq), G), dtype=np.float32)
        for i, idx in enumerate(_iter(idx_by_cl, "Cluster means")):
            n = max(len(idx), 1)
            M2[i, :] = M[idx, :].mean(axis=0) / n
        M = M2
        N = M.shape[0]
    else:
        bg_px = bg_py = None

    _log("[6/8] Computing lightening cutoffs...", verbose)
    if lighten_colors:
        group_index_arr = np.asarray(group_index)
        cutoffs = np.ones(len(gene_lists), dtype=np.float32)

        if lighten_colors_based_on_global:
            vals = M[M > 0]
            c = np.quantile(vals, 0.9) if vals.size else 1.0
            cutoffs[:] = max(float(c), 1e-12)
        else:
            for li in range(len(gene_lists)):
                vals = M[:, group_index_arr == li]
                vals = vals[vals > 0]
                c = np.quantile(vals, 0.9) if vals.size else 1.0
                cutoffs[li] = max(float(c), 1e-12)
    else:
        cutoffs = np.ones(len(gene_lists), dtype=np.float32)

    _log("[7/8] Rendering figure...", verbose)
    fig = plt.Figure(figsize=(W / dpi, H / dpi), dpi=dpi, facecolor="white")
    FigureCanvas(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.axis("off")

    if giant_pies and bg_px is not None:
        for i in _iter(range(len(bg_px)), "Drawing grey background cells"):
            ax.add_patch(Wedge((bg_px[i], bg_py[i]), r_base, 0, 360, facecolor="#CCCCCC", edgecolor="none"))

    group_sizes = np.asarray(group_sizes, dtype=np.float32)
    group_index_arr = np.asarray(group_index)
    color_bases = np.asarray(color_bases, dtype=object)

    n_colored = 0
    n_grey = 0

    for i in _iter(range(N), "Drawing pies"):
        row = M[i, :].copy()

        if cutoff_percentage:
            thr = float(row.sum() * cutoff_percentage / 100.0)
            row[row <= thr] = 0.0
        if cutoff_expression_value:
            row[row <= cutoff_expression_value] = 0.0
        if proportionalize:
            row = row / group_sizes

        nz = row > 0
        if not np.any(nz):
            ax.add_patch(Wedge((px[i], py[i]), r_base, 0, 360, facecolor="#CCCCCC", edgecolor="none"))
            n_grey += 1
            continue

        vals = row[nz]
        cols_idx = np.where(nz)[0]

        # colors
        cols = []
        for j, v in zip(cols_idx, vals):
            base = color_bases[j]
            if lighten_colors:
                li = group_index_arr[j]
                amt = float(np.clip(v / cutoffs[li], 0.0, 1.0))
            else:
                amt = 1.0
            cols.append(_lighten_color(base, amt))

        total = float(vals.sum())
        start = 0.0
        if giant_pies and idx_by_cl is not None:
            r = (r_base * np.sqrt(len(idx_by_cl[i]))) / 2.0
        else:
            r = r_base

        for v, c in zip(vals, cols):
            end = start + (v / total) * 360.0
            ax.add_patch(Wedge((px[i], py[i]), r, start, end, facecolor=c, edgecolor="none"))
            start = end

        n_colored += 1

    _log("[8/8] Saving image...", verbose)
    fig.savefig(output_filename, dpi=dpi, facecolor="white")
    plt.close(fig)

    dt = time.time() - t0
    _log(f"Done in {dt:.1f}s | colored={n_colored}, grey={n_grey}, cells_plotted={N}", verbose)

    return {
        "image_file": output_filename,
        "n_colored_pies": n_colored,
        "n_grey_pies": n_grey,
        "cells_plotted": int(N),
        "seconds": float(dt),
    }