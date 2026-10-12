"""Stability dimension selection for stabilized ICA."""

from __future__ import annotations

import numpy as np

from ._ica_utils import get_ica_entry, get_ica_matrix


def mstd(
    adata,
    m=2,
    M=30,
    step=2,
    n_runs=10,
    *,
    use_highly_variable=True,
    layer=None,
    fun=None,
    algorithm="picard_ext",
    max_iter=2000,
    n_jobs=-1,
    show_progress=True,
    random_state=0,
    key_added="ica",
):
    """Run stabilized ICA over a component-count range and store MSTD results.

    The returned mapping contains fitted sica estimators for immediate follow-up
    work. Only arrays and scalar metadata are saved to ``adata.uns`` so the
    AnnData remains writable as an ``.h5ad`` file.
    """
    import sica
    from sica.base import StabilizedICA

    if m < 1 or M < m or step < 1:
        raise ValueError("Require 1 <= m <= M and step >= 1.")
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1.")
    if fun is None:
        fun = "tanh" if "picard" in algorithm else "logcosh"

    matrix, _ = get_ica_matrix(
        adata, use_highly_variable=use_highly_variable, layer=layer
    )
    max_supported = min(matrix.shape[0] - 1, matrix.shape[1])
    if M > max_supported:
        raise ValueError(
            f"M={M} exceeds the maximum supported component count "
            f"({max_supported}) for matrix shape {matrix.shape}."
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
        matrix.T,
        n_components=int(component_counts[-1]),
        svd_solver="auto",
        chunked=False,
        chunk_size=None,
        zero_center=True,
        random_state=random_state,
    )

    fitted = {}
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
        fitted[int(count)] = estimator
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
            "fun": str(fun),
            "algorithm": str(algorithm),
            "max_iter": int(max_iter),
            "n_jobs": int(n_jobs),
            "random_state": int(random_state),
        },
    }
    adata.uns[key_added] = entry
    return fitted


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


def select_mstd_components(adata, last_n=10, *, key_added="ica"):
    """Suggest a dimension using the pooled-stability inflection heuristic.

    This reproduces the notebook heuristic: find the largest gap among pooled
    stability indexes from the highest tested dimensions, then choose the
    largest tested dimension whose median stability is at least that cutoff.
    Returns ``None`` when no tested dimension meets the cutoff.
    """
    entry = adata.uns.get(key_added)
    mstd_result = entry.get("mstd") if isinstance(entry, dict) else None
    if not isinstance(mstd_result, dict):
        raise KeyError(f"No MSTD results are stored in adata.uns[{key_added!r}].")
    if last_n < 1:
        raise ValueError("last_n must be at least 1.")

    component_counts = np.asarray(mstd_result["component_counts"], dtype=int)
    stability = np.asarray(mstd_result["stability_indexes"], dtype=np.float64)
    if stability.ndim != 2 or stability.shape[0] != len(component_counts):
        raise ValueError("Stored MSTD stability results have an invalid shape.")
    selected_rows = stability[-min(last_n, len(component_counts)) :]
    pooled = selected_rows[np.isfinite(selected_rows)]
    if pooled.size < 2:
        raise ValueError("At least two finite stability indexes are required.")
    ordered = np.sort(pooled)
    cutoff = float(ordered[np.argmax(np.diff(ordered))])
    medians = np.nanmedian(stability, axis=1)
    matching = np.flatnonzero(medians >= cutoff)

    mstd_result["inflection_threshold"] = cutoff
    if matching.size == 0:
        mstd_result["selected_n_components"] = -1
        adata.uns[key_added] = entry
        return None
    selected = int(component_counts[matching[-1]])
    mstd_result["selected_n_components"] = selected
    adata.uns[key_added] = entry
    return selected
