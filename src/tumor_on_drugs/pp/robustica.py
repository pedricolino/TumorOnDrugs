"""RobustICA for AnnData."""

from __future__ import annotations

import numpy as np

from ._ica_utils import (
    get_ica_entry,
    get_ica_matrix,
    orient_ica_matrix,
    store_string_array,
)


def _serializable_param(value):
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return _serializable_param(value.item())
    if isinstance(value, np.ndarray):
        if value.dtype.kind in "biufcSU":
            return value
        return repr(value)
    if isinstance(value, dict):
        return {str(key): _serializable_param(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable_param(item) for item in value]
    return repr(value)


def robustica(
    adata,
    n_comps=None,
    *,
    use_highly_variable=True,
    layer=None,
    use_pca=False,
    space="gene",
    center=True,
    key_added="ica",
    n_top_genes=30,
    verbose=True,
    copy=False,
    **robustica_kwargs,
):
    """Fit RobustICA and store scores, loadings, and serializable metadata.

    ``n_comps`` maps to RobustICA's ``n_components`` argument. Additional
    RobustICA estimator arguments, such as ``robust_runs``, ``robust_method``,
    and ``robust_kws``, can be passed as keyword arguments. RobustICA may return
    fewer components than requested; all stored results use the returned
    component count. ``verbose`` controls the parameter summary and the
    estimator's verbosity by default; a RobustICA ``verbose`` keyword overrides
    the latter. By default, genes are features and cells are observations.
    Set ``space="cell"`` to treat cells as features and genes as observations;
    this is incompatible with ``use_pca=True``. PCA-mode loadings are projected
    back to genes using ``adata.varm["PCs"]``.
    """
    from robustica import RobustICA

    if "n_components" in robustica_kwargs:
        raise TypeError("Use n_comps instead of RobustICA's n_components.")
    if n_comps is not None and (
        isinstance(n_comps, (bool, np.bool_))
        or not isinstance(n_comps, (int, np.integer))
        or n_comps < 1
    ):
        raise ValueError("n_comps must be a positive integer or None.")
    if n_top_genes < 1:
        raise ValueError("n_top_genes must be at least 1.")
    if use_pca and space != "gene":
        raise ValueError("use_pca=True is only supported with space='gene'.")

    ad = adata.copy() if copy else adata
    matrix, feature_mask = get_ica_matrix(
        ad,
        use_highly_variable=use_highly_variable,
        layer=layer,
        use_pca=use_pca,
        center=center,
    )
    oriented_matrix = orient_ica_matrix(matrix, space)
    if n_comps is not None:
        max_supported = min(oriented_matrix.shape)
        if n_comps > max_supported:
            raise ValueError(f"n_comps must be between 1 and {max_supported}.")

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

    estimator_params = {
        "n_components": n_comps,
        "verbose": verbose,
        **robustica_kwargs,
    }
    if verbose:
        print(
            "Running RobustICA with parameters: "
            f"{_serializable_param(estimator_params)}"
        )

    model = RobustICA(**estimator_params)
    scores, loadings = model.fit_transform(oriented_matrix)
    scores = np.asarray(scores, dtype=np.float64)
    loadings = np.asarray(loadings, dtype=np.float64)
    if (
        scores.ndim != 2
        or scores.shape[0] != oriented_matrix.shape[0]
        or scores.shape[1] < 1
    ):
        raise RuntimeError(
            "RobustICA returned score shape "
            f"{scores.shape}; expected ({oriented_matrix.shape[0]}, n_components) "
            "with at least one component."
        )
    if loadings.shape != (oriented_matrix.shape[1], scores.shape[1]):
        raise RuntimeError(
            f"RobustICA returned loading shape {loadings.shape}; expected "
            f"{(oriented_matrix.shape[1], scores.shape[1])}."
        )

    n_returned = scores.shape[1]
    if pca_components is not None:
        loadings = pca_components[:, : matrix.shape[1]] @ loadings
        loadings = loadings[feature_mask, :]

    component_names = [f"IC{i + 1}" for i in range(n_returned)]
    genes_used = ad.var_names[feature_mask]
    if space == "gene":
        cell_values = scores
        gene_values = loadings
    else:
        cell_values = loadings
        gene_values = scores
    full_gene_values = np.full(
        (ad.n_vars, n_returned), np.nan, dtype=np.float64
    )
    full_gene_values[feature_mask, :] = gene_values
    ad.obsm[f"X_{key_added}"] = cell_values
    ad.varm["ICs"] = full_gene_values

    top_count = min(n_top_genes, len(genes_used))
    gene_string_dtype = np.asarray(genes_used.astype(str)).dtype
    positive_genes = np.full((n_returned, top_count), "", dtype=gene_string_dtype)
    negative_genes = np.full((n_returned, top_count), "", dtype=gene_string_dtype)
    positive_values = np.full((n_returned, top_count), np.nan, dtype=np.float64)
    negative_values = np.full_like(positive_values, np.nan)
    for index in range(n_returned):
        order = np.argsort(gene_values[:, index])
        low = order[:top_count]
        high = order[-top_count:][::-1]
        positive_genes[index] = genes_used[high].astype(str)
        negative_genes[index] = genes_used[low].astype(str)
        positive_values[index] = gene_values[high, index]
        negative_values[index] = gene_values[low, index]

    params = {
        "method": "robustica",
        "n_comps": int(n_returned),
        "requested_n_comps": (
            None if n_comps is None else int(n_comps)
        ),
        "use_highly_variable": bool(use_highly_variable),
        "layer": "" if layer is None else str(layer),
        "use_pca": bool(use_pca),
        "space": space,
        "center": bool(center),
        "n_top_genes": int(n_top_genes),
        **{
            key: _serializable_param(value)
            for key, value in robustica_kwargs.items()
        },
    }
    entry = get_ica_entry(ad, key_added)
    entry.update({
        "params": params,
        "component_names": store_string_array(component_names),
        "genes_used": store_string_array(genes_used),
        "top_genes_positive": positive_genes,
        "top_genes_negative": negative_genes,
        "top_loadings_positive": positive_values,
        "top_loadings_negative": negative_values,
    })
    ad.uns[key_added] = entry
    return ad if copy else None
