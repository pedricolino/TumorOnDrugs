"""Matrix and AnnData preparation helpers."""

import numpy as np
import pandas as pd


def is_integerish_matrix(x, n_check: int = 50) -> bool:
    if hasattr(x, "X"):
        x = x.X
    if hasattr(x, "todense"):
        arr = x[:n_check, :n_check].todense()
    else:
        arr = np.asarray(x[:n_check, :n_check])
    try:
        return bool(np.all(np.mod(arr, 1) == 0))
    except Exception:
        return False


def ensure_raw_counts_in_X(adata):
    if is_integerish_matrix(adata.X):
        if not hasattr(adata, "raw") or adata.raw is None:
            print("No adata.raw found, but adata.X looks like raw counts. Setting adata.raw = adata.")
            adata.raw = adata.copy()
        return adata
    if hasattr(adata, "raw") and adata.raw is not None and is_integerish_matrix(adata.raw.X):
        var_idxs = [adata.raw.var_names.get_loc(gene) for gene in adata.var_names]
        obs_idxs = [adata.raw.obs_names.get_loc(cell) for cell in adata.obs_names]
        adata.X = adata.raw.X[obs_idxs, :][:, var_idxs].copy()
        return adata
    raise ValueError("Neither adata.X nor adata.raw.X appear to contain raw counts.")


def basic_checks(adata, input_h5ad: str, treatment_col: str, timepoint_col: str):
    for column in [treatment_col, timepoint_col]:
        if column not in adata.obs.columns:
            raise KeyError(f"Missing required obs column '{column}' in {input_h5ad}")

    adata.obs[timepoint_col] = pd.to_numeric(adata.obs[timepoint_col], errors="coerce")
    if adata.obs[timepoint_col].isna().any():
        bad = adata.obs[adata.obs[timepoint_col].isna()][[treatment_col]].head()
        raise ValueError(f"Some timepoints are NaN after numeric conversion. Example rows:\n{bad}")
