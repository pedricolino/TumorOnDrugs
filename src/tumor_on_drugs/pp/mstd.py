"""Stability dimension selection for stabilized ICA."""

from __future__ import annotations

import numpy as np

from ._ica_utils import get_ica_entry, get_ica_matrix, orient_ica_matrix


def mstd(
    adata,
    m=2,
    M=30,
    step=2,
    n_runs=10,
    *,
    use_highly_variable=True,
    layer=None,
    use_pca=False,
    space="gene",
    center=True,
    fun=None,
    algorithm="picard_ext",
    max_iter=2000,
    n_jobs=-1,
    show_progress=True,
    random_state=0,
    key_added="ica",
):
    """Run stabilized ICA over a component-count range and store MSTD results.

    Results are saved to ``adata.uns`` using arrays and scalar metadata so the
    AnnData remains writable as an ``.h5ad`` file. The progress bar is the only
    output; fitted estimators are not returned. Set ``space="cell"`` to treat
    cells as features and genes as observations. ``use_pca=True`` is only
    supported with the default ``space="gene"``. Set ``center=False`` to skip
    input centering.
    """
    import sica
    from sica.base import StabilizedICA

    if m < 1 or M < m or step < 1:
        raise ValueError("Require 1 <= m <= M and step >= 1.")
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1.")
    if fun is None:
        fun = "tanh" if "picard" in algorithm else "logcosh"
    if use_pca and space != "gene":
        raise ValueError("use_pca=True is only supported with space='gene'.")

    matrix, _ = get_ica_matrix(
        adata,
        use_highly_variable=use_highly_variable,
        layer=layer,
        use_pca=use_pca,
        center=center,
    )
    oriented_matrix = orient_ica_matrix(matrix, space)
    max_supported = min(oriented_matrix.shape[0] - 1, oriented_matrix.shape[1])
    if M > max_supported:
        raise ValueError(
            f"M={M} exceeds the maximum supported component count "
            f"({max_supported}) for matrix shape {oriented_matrix.shape}."
        )
    component_counts = np.arange(m, M + 1, step, dtype=int)
    if component_counts[-1] != M:
        component_counts = np.append(component_counts, M)
    if show_progress:
        try:
            from tqdm.auto import tqdm
        except ImportError as exc:
            raise ImportError(
                "mstd progress bars require tqdm; install the "
                "'stabilized-ica' extra or pass show_progress=False."
            ) from exc
        component_iterator = tqdm(
            component_counts, desc="MSTD", unit="component count"
        )
    else:
        component_iterator = component_counts
    whitened, _ = sica.base.whitening(
        oriented_matrix.T,
        n_components=int(component_counts[-1]),
        svd_solver="auto",
        chunked=False,
        chunk_size=None,
        zero_center=center,
        random_state=random_state,
    )

    stability_rows = []
    for count in component_iterator:
        estimator = StabilizedICA(
            n_components=int(count),
            n_runs=int(n_runs),
            algorithm=algorithm,
            fun=fun,
            whiten=False,
            max_iter=int(max_iter),
            n_jobs=int(n_jobs),
            plot=False,
        )
        estimator.fit(whitened[:, :count].T)
        stability = np.asarray(estimator.stability_indexes_, dtype=np.float64)
        if stability.ndim != 1 or stability.size != count:
            raise RuntimeError(
                "StabilizedICA returned an unexpected stability-index shape "
                f"{stability.shape} for {count} components."
            )
        stability_rows.append(stability)

    stored_stability = np.full((len(component_counts), M), np.nan, dtype=np.float64)
    for row, values in enumerate(stability_rows):
        stored_stability[row, : len(values)] = values
    entry = get_ica_entry(adata, key_added)
    entry["mstd"] = {
        "component_counts": component_counts,
        "stability_indexes": stored_stability,
        "mean_stability": np.asarray(
            [np.mean(values) for values in stability_rows], dtype=np.float64
        ),
        "median_stability": np.asarray(
            [np.median(values) for values in stability_rows], dtype=np.float64
        ),
        "params": {
            "m": int(m),
            "M": int(M),
            "step": int(step),
            "n_runs": int(n_runs),
            "use_highly_variable": bool(use_highly_variable),
            "layer": "" if layer is None else str(layer),
            "use_pca": bool(use_pca),
            "space": space,
            "center": bool(center),
            "fun": str(fun),
            "algorithm": str(algorithm),
            "max_iter": int(max_iter),
            "n_jobs": int(n_jobs),
            "random_state": int(random_state),
        },
    }
    adata.uns[key_added] = entry


def select_stable_components(adata, min_stability=0.8, *, key_added="ica"):
    """Return component indices whose final ICA stability meets a threshold."""
    entry = adata.uns.get(key_added)
    if not isinstance(entry, dict) or "stability_indexes" not in entry:
        raise KeyError(f"No final ICA stability indexes are stored in adata.uns[{key_added!r}].")
    stability = np.asarray(entry["stability_indexes"], dtype=np.float64)
    if stability.ndim != 1 or not np.isfinite(stability).all():
        raise ValueError("Stored stability_indexes must be a finite one-dimensional array.")
    if not 0 <= min_stability <= 1:
        raise ValueError("min_stability must be between 0 and 1.")
    return np.flatnonzero(stability >= min_stability) + 1


def select_mstd_components(adata, last_n=None, *, key_added="ica"):
    entry = adata.uns.get(key_added)
    mstd_result = entry.get("mstd") if isinstance(entry, dict) else None
    if not isinstance(mstd_result, dict):
        raise KeyError(f"No MSTD results are stored in adata.uns[{key_added!r}].")
    if last_n is None:
        last_n = len(mstd_result["component_counts"])
    if last_n < 1:
        raise ValueError("last_n must be at least 1.")

    component_counts = np.asarray(mstd_result["component_counts"], dtype=int)
    stability = np.asarray(mstd_result["stability_indexes"], dtype=np.float64)

    if stability.ndim != 2 or stability.shape[0] != len(component_counts):
        raise ValueError("Stored MSTD stability results have an invalid shape.")

    order = np.argsort(component_counts)
    component_counts = component_counts[order]
    stability = stability[order]

    medians = np.nanmedian(stability, axis=1)

    # Use only the tail of the curve if that's what your heuristic intends
    tail = medians[-min(last_n, len(medians)):]
    x = component_counts[-len(tail):]

    if len(tail) < 2:
        raise ValueError("At least two stability points are required.")

    # Elbow = point with largest drop in the median curve
    drops = -np.diff(tail)
    elbow_idx = int(np.argmax(drops))  # index before largest drop
    selected = int(x[elbow_idx])

    mstd_result["optimal_n_compt"] = selected
    mstd_result["selected_n_components"] = selected
    mstd_result["selected_median_stability"] = float(tail[elbow_idx])
    # useful for plotting the elbow:
    mstd_result["selected_stability_drop"] = float(drops[elbow_idx])
    adata.uns[key_added] = entry
    return selected
