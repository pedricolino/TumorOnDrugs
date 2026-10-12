"""Shared matrix and metadata helpers for ICA routines."""

from __future__ import annotations

import numpy as np
from scipy import sparse


def _is_sparse_matrix(matrix):
    return sparse.issparse(matrix) or (
        hasattr(matrix, "format") and callable(getattr(matrix, "toarray", None))
    )


def _to_numpy(matrix):
    if _is_sparse_matrix(matrix):
        matrix = matrix.toarray()
    get = getattr(matrix, "get", None)
    if callable(get):
        matrix = get()
    return matrix


def get_ica_matrix(adata, *, use_highly_variable=True, layer=None):
    if layer is not None:
        if layer not in adata.layers:
            raise KeyError(f"Layer {layer!r} was not found in adata.layers.")
        matrix = adata.layers[layer]
    else:
        matrix = adata.X

    if not sparse.issparse(matrix):
        get = getattr(matrix, "get", None)
        if callable(get):
            matrix = get()

    if use_highly_variable and "highly_variable" in adata.var:
        feature_mask = adata.var["highly_variable"].to_numpy(dtype=bool)
        if not feature_mask.any():
            raise ValueError("No genes are marked as highly_variable=True.")
    else:
        feature_mask = np.ones(adata.n_vars, dtype=bool)
    matrix = matrix[:, feature_mask]
    matrix = _to_numpy(matrix)
    matrix = np.asarray(matrix)
    if matrix.dtype == object and any(
        _is_sparse_matrix(value) for value in matrix.flat
    ):
        dense_matrix = np.empty(matrix.shape, dtype=object)
        for index, value in np.ndenumerate(matrix):
            if _is_sparse_matrix(value):
                value = _to_numpy(value)
                if value.shape != (1, 1):
                    raise ValueError(
                        "ICA input contains a non-scalar sparse value at "
                        f"{index} with shape {value.shape}."
                    )
                dense_matrix[index] = value[0, 0]
            else:
                dense_matrix[index] = value
        matrix = dense_matrix
    matrix = np.asarray(matrix, dtype=np.float64).copy()
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("ICA input must be a finite two-dimensional matrix.")
    matrix -= matrix.mean(axis=0, keepdims=True)
    return matrix, feature_mask


def get_ica_entry(adata, key_added):
    value = adata.uns.get(key_added)
    return dict(value) if isinstance(value, dict) else {}


def store_string_array(values):
    return np.asarray(values, dtype=str)
