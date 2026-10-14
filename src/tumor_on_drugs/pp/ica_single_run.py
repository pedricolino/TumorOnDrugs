"""Conventional single-fit FastICA for AnnData."""

import numpy as np
import pandas as pd

from ._ica_utils import (
    get_ica_entry,
    get_ica_matrix,
    orient_ica_matrix,
    store_string_array,
)


def ica_single_run(
    adata,
    n_comps=30,
    use_highly_variable=True,
    layer=None,
    space="gene",
    key_added="ica",
    random_state=0,
    max_iter=500,
    tol=1e-4,
    n_top_genes=20,
    copy=False,
):
    """Fit FastICA once, treating genes or cells as ICA features."""
    from sklearn.decomposition import FastICA

    ad = adata.copy() if copy else adata
    matrix, feature_mask = get_ica_matrix(
        ad, use_highly_variable=use_highly_variable, layer=layer
    )
    oriented_matrix = orient_ica_matrix(matrix, space)
    if n_comps < 1 or n_comps > min(oriented_matrix.shape):
        raise ValueError(
            f"n_comps must be between 1 and {min(oriented_matrix.shape)}."
        )
    if n_top_genes < 1:
        raise ValueError("n_top_genes must be at least 1.")

    model = FastICA(
        n_components=n_comps,
        random_state=random_state,
        max_iter=max_iter,
        tol=tol,
    )
    scores = model.fit_transform(oriented_matrix)
    loadings = model.components_.T
    genes_used = ad.var_names[feature_mask]
    if space == "gene":
        cell_values = scores
        gene_values = loadings
    else:
        cell_values = loadings
        gene_values = scores
    component_names = [f"IC{i + 1}" for i in range(n_comps)]
    ad.obsm[f"X_{key_added}"] = cell_values
    full_gene_values = np.full((ad.n_vars, n_comps), np.nan, dtype=np.float64)
    full_gene_values[feature_mask, :] = gene_values
    ad.varm[f"{key_added}_components"] = full_gene_values

    positive = np.full((n_comps, n_top_genes), "", dtype=object)
    negative = np.full_like(positive, "")
    positive_values = np.full((n_comps, n_top_genes), np.nan)
    negative_values = np.full_like(positive_values, np.nan)
    for index in range(n_comps):
        values = pd.Series(gene_values[:, index], index=genes_used)
        top = values.nlargest(min(n_top_genes, len(values)))
        bottom = values.nsmallest(min(n_top_genes, len(values)))
        positive[index, : len(top)] = top.index.astype(str)
        positive_values[index, : len(top)] = top.to_numpy()
        negative[index, : len(bottom)] = bottom.index.astype(str)
        negative_values[index, : len(bottom)] = bottom.to_numpy()

    entry = get_ica_entry(ad, key_added)
    entry.update({
        "params": {
            "method": "fastica_single_run",
            "n_comps": int(n_comps),
            "use_highly_variable": bool(use_highly_variable),
            "layer": "" if layer is None else str(layer),
            "space": space,
            "random_state": int(random_state),
            "max_iter": int(max_iter),
            "tol": float(tol),
            "n_top_genes": int(n_top_genes),
        },
        "component_names": store_string_array(component_names),
        "genes_used": store_string_array(genes_used),
        "top_genes_positive": positive.astype(str),
        "top_genes_negative": negative.astype(str),
        "top_loadings_positive": positive_values,
        "top_loadings_negative": negative_values,
    })
    ad.uns[key_added] = entry
    return ad if copy else None
