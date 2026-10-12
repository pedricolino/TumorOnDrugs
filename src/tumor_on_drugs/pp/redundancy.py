"""Component clustering and top-gene redundancy summaries."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from ._ica_utils import get_ica_entry


def component_gene_redundancy(
    adata,
    top_n=100,
    *,
    key_added="ica",
    components_key="ICs",
):
    """Measure pairwise overlap of the top absolute-loading genes per IC."""
    if components_key not in adata.varm:
        raise KeyError(f"Loadings {components_key!r} were not found in adata.varm.")
    if top_n < 1:
        raise ValueError("top_n must be at least 1.")

    loadings = np.asarray(adata.varm[components_key], dtype=np.float64)
    if loadings.ndim != 2 or loadings.shape[0] != adata.n_vars:
        raise ValueError("Component loadings must have one row per gene.")
    n_components = loadings.shape[1]
    gene_sets = []
    for component in range(n_components):
        valid = np.isfinite(loadings[:, component])
        gene_indices = np.flatnonzero(valid)
        order = np.argsort(np.abs(loadings[valid, component]))[::-1][:top_n]
        gene_sets.append(set(map(str, adata.var_names[gene_indices[order]])))

    rows = []
    for first in range(n_components):
        for second in range(first + 1, n_components):
            shared = sorted(gene_sets[first] & gene_sets[second])
            union = gene_sets[first] | gene_sets[second]
            rows.append({
                "component_a": f"IC{first + 1}",
                "component_b": f"IC{second + 1}",
                "n_shared": len(shared),
                "jaccard": len(shared) / len(union) if union else 0.0,
                "shared_genes": ", ".join(shared),
            })
    result = pd.DataFrame(
        rows,
        columns=["component_a", "component_b", "n_shared", "jaccard", "shared_genes"],
    )

    entry = get_ica_entry(adata, key_added)
    entry["gene_redundancy"] = {
        "top_n": int(top_n),
        "component_a": result["component_a"].to_numpy(dtype=str),
        "component_b": result["component_b"].to_numpy(dtype=str),
        "n_shared": result["n_shared"].to_numpy(dtype=np.int64),
        "jaccard": result["jaccard"].to_numpy(dtype=np.float64),
        "shared_genes": result["shared_genes"].to_numpy(dtype=str),
    }
    adata.uns[key_added] = entry
    return result


def cluster_components(
    adata,
    correlation_threshold=0.8,
    *,
    embedding_key="X_ica",
    key_added="ica",
):
    """Cluster components by sign-invariant Spearman correlation of cell scores."""
    if embedding_key not in adata.obsm:
        raise KeyError(f"Embedding {embedding_key!r} was not found in adata.obsm.")
    if not 0 < correlation_threshold <= 1:
        raise ValueError("correlation_threshold must be in (0, 1].")

    scores = np.asarray(adata.obsm[embedding_key], dtype=np.float64)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("Component scores must be a finite two-dimensional matrix.")
    n_components = scores.shape[1]
    if n_components == 0:
        raise ValueError("The component embedding contains no components.")
    if n_components == 1:
        labels = np.ones(1, dtype=np.int64)
    else:
        if n_components == 2:
            pair_correlation = float(
                spearmanr(scores[:, 0], scores[:, 1]).statistic
            )
            correlation = np.array(
                [[1.0, pair_correlation], [pair_correlation, 1.0]],
                dtype=np.float64,
            )
        else:
            correlation = np.asarray(
                spearmanr(scores, axis=0).statistic, dtype=np.float64
            )
        correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
        similarity = np.clip(np.abs(correlation), 0.0, 1.0)
        np.fill_diagonal(similarity, 1.0)
        distance = 1.0 - similarity
        np.fill_diagonal(distance, 0.0)
        tree = linkage(squareform(distance, checks=False), method="average")
        labels = fcluster(tree, t=1.0 - correlation_threshold, criterion="distance")
        labels = labels.astype(np.int64)

    component_names = [f"IC{i + 1}" for i in range(n_components)]
    result = pd.Series(labels, index=component_names, name="cluster")
    entry = get_ica_entry(adata, key_added)
    entry["component_clusters"] = {
        "component_names": np.asarray(component_names, dtype=str),
        "labels": labels,
        "correlation_threshold": float(correlation_threshold),
        "method": "absolute_spearman_average_linkage",
    }
    adata.uns[key_added] = entry
    return result
