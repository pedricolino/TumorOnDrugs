"""Stabilized ICA for AnnData."""

from __future__ import annotations

import numpy as np

from ._ica_utils import (
    get_ica_entry,
    get_ica_matrix,
    orient_ica_matrix,
    store_string_array,
)


def stabilized_ica(
    adata,
    n_comps=None,
    n_runs=30,
    *,
    optimal_n_compt=None,
    use_highly_variable=True,
    layer=None,
    use_pca=False,
    space="gene",
    center=True,
    whiten=True,
    key_added="ica",
    algorithm="picard_ext",
    fun=None,
    max_iter=2000,
    n_jobs=1,
    n_top_genes=30,
    verbose=True,
    copy=False,
):
    """Fit stabilized ICA and store its scores, loadings, and serializable metadata.

    If ``n_comps`` is omitted, use ``optimal_n_compt`` or the elbow selected
    from stored MSTD results. An explicit ``n_comps`` overrides the elbow.
    By default, ICA treats genes as features and cells as observations. Set
    ``space="cell"`` to treat cells as features and genes as observations;
    this is incompatible with ``use_pca=True``. PCA-mode gene loadings are
    projected back using ``adata.varm["PCs"]``. In either space, cell-aligned
    component values are stored in ``adata.obsm['X_<key_added>']`` and
    gene-aligned values in ``adata.varm['ICs']``. If ``whiten=False``, the
    input is explicitly whitened to ``n_comps`` dimensions before fitting.
    No estimator object is stored in ``adata.uns``.
    """
    from sica.base import StabilizedICA, whitening

    ad = adata.copy() if copy else adata
    entry = ad.uns.get(key_added)
    mstd_result = entry.get("mstd") if isinstance(entry, dict) else None
    stored_optimal = None
    mstd_params = (
        mstd_result.get("params", {}) if isinstance(mstd_result, dict) else {}
    )
    mstd_space = mstd_params.get("space", "gene")
    if isinstance(mstd_result, dict):
        if mstd_space == space:
            stored_optimal = mstd_result.get(
                "optimal_n_compt", mstd_result.get("selected_n_components")
            )
    if optimal_n_compt is None:
        optimal_n_compt = stored_optimal
    if n_comps is None:
        if optimal_n_compt is None:
            if not isinstance(mstd_result, dict):
                raise ValueError(
                    "n_comps=None requires MSTD results; run pp.mstd first or "
                    "provide n_comps explicitly."
                )
            if mstd_space != space:
                raise ValueError(
                    f"Stored MSTD results use space={mstd_space!r}; run "
                    f"pp.mstd with space={space!r} or provide n_comps."
                )
            from .mstd import select_mstd_components

            optimal_n_compt = select_mstd_components(ad, key_added=key_added)
        n_comps = optimal_n_compt

    if optimal_n_compt is not None:
        if (
            isinstance(optimal_n_compt, (bool, np.bool_))
            or not isinstance(optimal_n_compt, (int, np.integer))
            or optimal_n_compt < 1
        ):
            raise ValueError("optimal_n_compt must be a positive integer or None.")
        optimal_n_compt = int(optimal_n_compt)

    if use_pca and space != "gene":
        raise ValueError("use_pca=True is only supported with space='gene'.")
    matrix, feature_mask = get_ica_matrix(
        ad,
        use_highly_variable=use_highly_variable,
        layer=layer,
        use_pca=use_pca,
        center=center,
    )
    oriented_matrix = orient_ica_matrix(matrix, space)
    pca_components = None
    if use_pca:
        if "PCs" not in ad.varm:
            raise KeyError("PCA loadings 'PCs' were not found in adata.varm.")
        pca_components = np.asarray(ad.varm["PCs"], dtype=np.float64)
        if (
            pca_components.ndim != 2
            or pca_components.shape[0] != ad.n_vars
            or pca_components.shape[1] < matrix.shape[1]
            or not np.isfinite(pca_components[:, : matrix.shape[1]]).all()
        ):
            raise ValueError(
                "PCA loadings in adata.varm['PCs'] must have one row per gene "
                "and at least one column per X_pca component."
            )
    max_supported = min(oriented_matrix.shape[0] - 1, oriented_matrix.shape[1])
    if n_comps < 1 or n_comps > max_supported:
        raise ValueError(f"n_comps must be between 1 and {max_supported}.")
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1.")
    if n_top_genes < 1:
        raise ValueError("n_top_genes must be at least 1.")
    if fun is None:
        fun = "tanh" if "picard" in algorithm else "logcosh"

    params = {
        "n_comps": int(n_comps),
        "optimal_n_compt": optimal_n_compt,
        "n_runs": int(n_runs),
        "use_highly_variable": bool(use_highly_variable),
        "layer": "" if layer is None else str(layer),
        "use_pca": bool(use_pca),
        "space": space,
        "center": bool(center),
        "whiten": bool(whiten),
        "algorithm": str(algorithm),
        "fun": str(fun),
        "max_iter": int(max_iter),
        "n_jobs": int(n_jobs),
        "n_top_genes": int(n_top_genes),
    }
    if verbose:
        print(f"Running stabilized ICA with parameters: {params}")

    fit_matrix = oriented_matrix.T
    estimator_whiten = whiten
    if not whiten:
        fit_matrix, _ = whitening(
            oriented_matrix,
            n_components=int(n_comps),
            svd_solver="auto",
            chunked=False,
            chunk_size=None,
            zero_center=center,
        )
        fit_matrix = fit_matrix.T

    model = StabilizedICA(
        n_components=int(n_comps),
        n_runs=int(n_runs),
        algorithm=algorithm,
        fun=fun,
        whiten=estimator_whiten,
        max_iter=int(max_iter),
        n_jobs=int(n_jobs),
        plot=False,
    )
    model.fit(fit_matrix)
    scores = np.asarray(model.S_, dtype=np.float64).T
    if scores.shape != (oriented_matrix.shape[0], n_comps):
        raise RuntimeError(
            f"StabilizedICA returned score shape {scores.shape}; expected "
            f"{(oriented_matrix.shape[0], n_comps)}."
        )

    raw_loadings = np.asarray(model.transform(oriented_matrix.T), dtype=np.float64)
    if raw_loadings.shape == (n_comps, oriented_matrix.shape[1]):
        loadings = raw_loadings.T
    elif raw_loadings.shape == (oriented_matrix.shape[1], n_comps):
        loadings = raw_loadings
    else:
        raise RuntimeError(
            f"StabilizedICA returned loading shape {raw_loadings.shape}; "
            f"expected {(n_comps, oriented_matrix.shape[1])} or "
            f"{(oriented_matrix.shape[1], n_comps)}."
        )
    if pca_components is not None:
        loadings = pca_components[:, : matrix.shape[1]] @ loadings
        loadings = loadings[feature_mask, :]

    component_names = [f"IC{i + 1}" for i in range(n_comps)]
    genes_used = ad.var_names[feature_mask]
    if space == "gene":
        cell_values = scores
        gene_values = loadings
    else:
        cell_values = loadings
        gene_values = scores
    full_gene_values = np.full((ad.n_vars, n_comps), np.nan, dtype=np.float64)
    full_gene_values[feature_mask, :] = gene_values
    ad.obsm[f"X_{key_added}"] = cell_values
    ad.varm["ICs"] = full_gene_values

    top_count = min(n_top_genes, len(genes_used))
    gene_string_dtype = np.asarray(genes_used.astype(str)).dtype
    positive_genes = np.full((n_comps, top_count), "", dtype=gene_string_dtype)
    negative_genes = np.full((n_comps, top_count), "", dtype=gene_string_dtype)
    positive_values = np.full((n_comps, top_count), np.nan, dtype=np.float64)
    negative_values = np.full_like(positive_values, np.nan)
    for index in range(n_comps):
        order = np.argsort(gene_values[:, index])
        low = order[:top_count]
        high = order[-top_count:][::-1]
        positive_genes[index] = genes_used[high].astype(str)
        negative_genes[index] = genes_used[low].astype(str)
        positive_values[index] = gene_values[high, index]
        negative_values[index] = gene_values[low, index]

    entry = get_ica_entry(ad, key_added)
    entry.update({
        "params": {"method": "stabilized_ica", **params},
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
