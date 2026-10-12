"""Stabilized ICA for AnnData."""

from __future__ import annotations

import numpy as np

from ._ica_utils import get_ica_entry, get_ica_matrix, store_string_array


def ica(
    adata,
    n_comps=20,
    n_runs=30,
    *,
    use_highly_variable=True,
    layer=None,
    key_added="ica",
    algorithm="picard_ext",
    fun=None,
    max_iter=2000,
    n_jobs=1,
    n_top_genes=30,
    copy=False,
):
    """Fit stabilized ICA and store its scores, loadings, and serializable metadata.

    Scores are written to ``adata.obsm['X_<key_added>']`` and gene loadings to
    ``adata.varm['ICs']``. No estimator object is stored in ``adata.uns``.
    """
    from sica.base import StabilizedICA

    ad = adata.copy() if copy else adata
    matrix, feature_mask = get_ica_matrix(
        ad, use_highly_variable=use_highly_variable, layer=layer
    )
    max_supported = min(matrix.shape[0] - 1, matrix.shape[1])
    if n_comps < 1 or n_comps > max_supported:
        raise ValueError(f"n_comps must be between 1 and {max_supported}.")
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1.")
    if n_top_genes < 1:
        raise ValueError("n_top_genes must be at least 1.")
    if fun is None:
        fun = "tanh" if "picard" in algorithm else "logcosh"

    model = StabilizedICA(
        n_components=int(n_comps),
        n_runs=int(n_runs),
        algorithm=algorithm,
        fun=fun,
        whiten=True,
        max_iter=int(max_iter),
        n_jobs=int(n_jobs),
        plot=False,
    )
    model.fit(matrix)
    scores = np.asarray(model.transform(matrix), dtype=np.float64)
    if scores.shape != (ad.n_obs, n_comps):
        raise RuntimeError(
            f"StabilizedICA returned score shape {scores.shape}; expected "
            f"{(ad.n_obs, n_comps)}."
        )

    raw_loadings = np.asarray(model.S_, dtype=np.float64)
    if raw_loadings.shape == (n_comps, matrix.shape[1]):
        loadings = raw_loadings.T
    elif raw_loadings.shape == (matrix.shape[1], n_comps):
        loadings = raw_loadings
    else:
        raise RuntimeError(
            f"StabilizedICA returned loading shape {raw_loadings.shape}; "
            f"expected {(n_comps, matrix.shape[1])} or "
            f"{(matrix.shape[1], n_comps)}."
        )

    component_names = [f"IC{i + 1}" for i in range(n_comps)]
    genes_used = ad.var_names[feature_mask]
    full_loadings = np.full((ad.n_vars, n_comps), np.nan, dtype=np.float64)
    full_loadings[feature_mask, :] = loadings
    ad.obsm[f"X_{key_added}"] = scores
    ad.varm["ICs"] = full_loadings

    top_count = min(n_top_genes, len(genes_used))
    gene_string_dtype = np.asarray(genes_used.astype(str)).dtype
    positive_genes = np.full((n_comps, top_count), "", dtype=gene_string_dtype)
    negative_genes = np.full((n_comps, top_count), "", dtype=gene_string_dtype)
    positive_values = np.full((n_comps, top_count), np.nan, dtype=np.float64)
    negative_values = np.full_like(positive_values, np.nan)
    for index in range(n_comps):
        order = np.argsort(loadings[:, index])
        low = order[:top_count]
        high = order[-top_count:][::-1]
        positive_genes[index] = genes_used[high].astype(str)
        negative_genes[index] = genes_used[low].astype(str)
        positive_values[index] = loadings[high, index]
        negative_values[index] = loadings[low, index]

    entry = get_ica_entry(ad, key_added)
    entry.update({
        "params": {
            "method": "stabilized_ica",
            "n_comps": int(n_comps),
            "n_runs": int(n_runs),
            "use_highly_variable": bool(use_highly_variable),
            "layer": "" if layer is None else str(layer),
            "algorithm": str(algorithm),
            "fun": str(fun),
            "max_iter": int(max_iter),
            "n_jobs": int(n_jobs),
            "n_top_genes": int(n_top_genes),
        },
        "component_names": store_string_array(component_names),
        "genes_used": store_string_array(genes_used),
        "stability_indexes": np.asarray(
            model.stability_indexes_, dtype=np.float64
        ),
        "top_genes_positive": positive_genes,
        "top_genes_negative": negative_genes,
        "top_loadings_positive": positive_values,
        "top_loadings_negative": negative_values,
    })
    ad.uns[key_added] = entry
    return ad if copy else None
