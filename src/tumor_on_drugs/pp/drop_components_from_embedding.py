"""Embedding postprocessing helpers."""

import numpy as np


def drop_components_from_embedding(
    adata,
    component_indices,
    *,
    base_obsm_key="X_pca",
    pca_uns_key="pca",
    base_varm_key="PCs",
    new_suffix=None,
    assume_one_indexed=True,
):
    """Create derived slots with selected embedding components removed."""
    if base_obsm_key not in adata.obsm:
        raise ValueError(f"'{base_obsm_key}' not found in adata.obsm; run PCA first.")
    if component_indices is None or len(component_indices) == 0:
        raise ValueError("component_indices must contain at least one component to drop.")

    base_embed = adata.obsm[base_obsm_key]
    n_pcs = base_embed.shape[1]
    indices = np.asarray(component_indices, dtype=int)
    if assume_one_indexed:
        indices = indices - 1
    if (indices < 0).any() or (indices >= n_pcs).any():
        raise ValueError("component_indices contain an unavailable component.")
    indices = np.unique(indices)
    keep = np.setdiff1d(np.arange(n_pcs), indices, assume_unique=False)
    if keep.size == 0:
        raise ValueError("Removing all PCs is not allowed.")

    if new_suffix is None:
        labels = "_".join(str(int(i + (1 if assume_one_indexed else 0))) for i in indices)
        new_suffix = f"no_components_{labels}"

    new_obsm_key = f"{base_obsm_key}_{new_suffix}"
    adata.obsm[new_obsm_key] = base_embed[:, keep]
    variance_ratio_key = None
    if pca_uns_key in adata.uns:
        pca_entry = adata.uns[pca_uns_key]
        if isinstance(pca_entry, dict) and "variance_ratio" in pca_entry:
            variance_ratio_key = f"variance_ratio_{new_suffix}"
            pca_entry[variance_ratio_key] = np.asarray(pca_entry["variance_ratio"])[keep]

    varm_key = None
    if base_varm_key in adata.varm:
        varm_key = f"{base_varm_key}_{new_suffix}"
        adata.varm[varm_key] = np.asarray(adata.varm[base_varm_key])[:, keep]
    return {"obsm": new_obsm_key, "varm": varm_key, "variance_ratio": variance_ratio_key}
