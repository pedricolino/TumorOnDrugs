"""Compute score associations with PCA or ICA components."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from ._ica_utils import get_ica_entry, store_string_array


def compute_score_component_associations(
    adata,
    score_key,
    *,
    embedding_key="X_ica",
    method="spearman",
    compute_mutual_info=True,
    n_neighbors=3,
    random_state=0,
    key_added="ica",
):
    """Compute component-wise correlations and optional mutual information.

    Results are returned as a DataFrame and persisted as arrays under
    ``adata.uns[key_added]['associations'][score_key]`` for H5AD compatibility.
    """
    if score_key not in adata.obs:
        raise KeyError(f"Score {score_key!r} was not found in adata.obs.")
    if embedding_key not in adata.obsm:
        raise KeyError(f"Embedding {embedding_key!r} was not found in adata.obsm.")
    if method not in {"pearson", "spearman"}:
        raise ValueError("method must be either 'pearson' or 'spearman'.")
    if n_neighbors < 1:
        raise ValueError("n_neighbors must be at least 1.")

    scores = np.asarray(adata.obsm[embedding_key], dtype=np.float64)
    target = pd.to_numeric(adata.obs[score_key], errors="coerce").to_numpy(dtype=float)
    if scores.ndim != 2 or scores.shape[0] != len(target):
        raise ValueError("The embedding and score must have the same observation count.")
    if not np.isfinite(scores).all() or not np.isfinite(target).all():
        raise ValueError("Embedding and score values must all be finite.")
    if len(target) < 2:
        raise ValueError("At least two observations are required for association analysis.")

    correlation_fn = stats.pearsonr if method == "pearson" else stats.spearmanr
    correlations = np.empty(scores.shape[1], dtype=np.float64)
    p_values = np.empty(scores.shape[1], dtype=np.float64)
    mutual_info = np.empty(scores.shape[1], dtype=np.float64) if compute_mutual_info else None
    if compute_mutual_info:
        from sklearn.feature_selection import mutual_info_regression

    for index in range(scores.shape[1]):
        result = correlation_fn(scores[:, index], target)
        correlations[index] = float(result.statistic)
        p_values[index] = float(result.pvalue)
        if compute_mutual_info:
            mutual_info[index] = mutual_info_regression(
                scores[:, index].reshape(-1, 1),
                target,
                n_neighbors=min(n_neighbors, max(1, len(target) - 1)),
                random_state=random_state,
            )[0]

    component_label = "PC" if "pca" in embedding_key.lower() else "IC"
    component_names = [
        f"{component_label}{i + 1}" for i in range(scores.shape[1])
    ]
    frame = pd.DataFrame({
        "pc_index": np.arange(1, scores.shape[1] + 1, dtype=int),
        "component": component_names,
        "correlation": correlations,
        "abs_correlation": np.abs(correlations),
        "p_value": p_values,
    })
    if mutual_info is not None:
        frame["mutual_info"] = mutual_info

    entry = get_ica_entry(adata, key_added)
    associations = dict(entry.get("associations", {}))
    stored = {
        "embedding_key": str(embedding_key),
        "method": str(method),
        "component_names": store_string_array(component_names),
        "pc_index": frame["pc_index"].to_numpy(dtype=np.int64),
        "correlation": correlations,
        "abs_correlation": np.abs(correlations),
        "p_value": p_values,
    }
    if mutual_info is not None:
        stored["mutual_info"] = mutual_info
    associations[str(score_key)] = stored
    entry["associations"] = associations
    adata.uns[key_added] = entry
    return frame
