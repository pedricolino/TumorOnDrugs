"""Embedding postprocessing helpers."""

import numpy as np


def drop_components_from_embedding(
    adata,
    component_indices=None,
    *,
    correlation_threshold=None,
    mutual_info_threshold=None,
    score_keys=None,
    key_added="ica",
    base_obsm_key="X_ica",
    pca_uns_key="ica",
    base_varm_key="ICs",
    new_suffix=None,
    assume_one_indexed=True,
):
    """Create derived slots with selected ICA or embedding components removed.

    Provide component indices directly, or omit them to select components from
    stored score-association thresholds. If both cutoff arguments are omitted,
    saved plotting thresholds are used; otherwise, only the explicitly supplied
    cutoffs are used. Correlation and mutual-information cutoffs are combined
    with OR; MI is not used unless a cutoff is set.
    """
    if base_obsm_key not in adata.obsm:
        raise ValueError(f"'{base_obsm_key}' not found in adata.obsm.")
    base_embed = adata.obsm[base_obsm_key]
    n_components = base_embed.shape[1]

    if component_indices is None:

        def validate_threshold(value, name, *, maximum=None):
            if value is None:
                return None
            if isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be a finite non-negative number or None.")
            try:
                value = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{name} must be a finite non-negative number or None."
                ) from exc
            if not np.isfinite(value) or value < 0 or (
                maximum is not None and value > maximum
            ):
                if maximum is None:
                    expectation = "finite and non-negative"
                else:
                    expectation = f"between 0 and {maximum:g}"
                raise ValueError(f"{name} must be {expectation}, or None.")
            return value

        correlation_threshold = validate_threshold(
            correlation_threshold, "correlation_threshold", maximum=1
        )
        mutual_info_threshold = validate_threshold(
            mutual_info_threshold, "mutual_info_threshold"
        )
        use_stored_thresholds = (
            correlation_threshold is None and mutual_info_threshold is None
        )

        entry = adata.uns.get(key_added)
        associations = entry.get("associations") if isinstance(entry, dict) else None
        if not isinstance(associations, dict) or not associations:
            raise KeyError(
                f"No score associations found in adata.uns[{key_added!r}]; "
                "compute and plot associations first or provide component_indices."
            )
        if score_keys is None:
            selected_scores = list(associations)
        elif isinstance(score_keys, str):
            selected_scores = [score_keys]
        else:
            selected_scores = list(score_keys)
        if not selected_scores:
            raise ValueError("score_keys must contain at least one score.")

        selected_components = set()
        active_threshold_found = False
        for score_key in selected_scores:
            if score_key not in associations:
                raise KeyError(f"No associations for score {score_key!r} are stored.")
            association = associations[score_key]
            thresholds = association.get("thresholds", {})
            corr_cutoff = (
                thresholds.get("correlation_threshold")
                if use_stored_thresholds
                else correlation_threshold
            )
            mi_cutoff = (
                thresholds.get("mutual_info_threshold")
                if use_stored_thresholds
                else mutual_info_threshold
            )
            if corr_cutoff is None and mi_cutoff is None:
                continue
            active_threshold_found = True

            association_embedding = association.get("embedding_key", base_obsm_key)
            if association_embedding != base_obsm_key:
                raise ValueError(
                    f"Associations for {score_key!r} use embedding "
                    f"{association_embedding!r}, not {base_obsm_key!r}."
                )
            indices_for_score = np.asarray(association["pc_index"], dtype=int)
            if (
                indices_for_score.ndim != 1
                or len(indices_for_score) != n_components
                or (indices_for_score < 1).any()
                or (indices_for_score > n_components).any()
            ):
                raise ValueError(
                    f"Stored component indices for {score_key!r} do not match "
                    f"embedding {base_obsm_key!r}."
                )
            matches = np.zeros(len(indices_for_score), dtype=bool)
            if corr_cutoff is not None:
                corr = np.asarray(
                    association.get(
                        "abs_correlation",
                        np.abs(association["correlation"]),
                    ),
                    dtype=float,
                )
                if corr.shape != indices_for_score.shape:
                    raise ValueError(
                        f"Stored correlations for {score_key!r} have an invalid shape."
                    )
                matches |= corr >= corr_cutoff
            if mi_cutoff is not None:
                if "mutual_info" not in association:
                    raise ValueError(
                        f"No mutual-information values are stored for "
                        f"{score_key!r}; compute them before using an MI cutoff."
                    )
                mutual_info = np.asarray(association["mutual_info"], dtype=float)
                if mutual_info.shape != indices_for_score.shape:
                    raise ValueError(
                        f"Stored mutual-information values for {score_key!r} "
                        "have an invalid shape."
                    )
                matches |= mutual_info >= mi_cutoff
            selected_components.update(indices_for_score[matches].tolist())

        if not active_threshold_found:
            raise ValueError(
                "No saved cutoffs were found for the selected scores; plot the "
                "associations first or provide a correlation_threshold or "
                "mutual_info_threshold."
            )
        component_indices = sorted(selected_components)
        if not component_indices:
            raise ValueError(
                "No components met the selected association thresholds."
            )
    elif correlation_threshold is not None or mutual_info_threshold is not None:
        raise ValueError(
            "Pass component_indices or association cutoffs, not both."
        )
    elif score_keys is not None:
        raise ValueError("Pass component_indices or score_keys, not both.")

    if np.isscalar(component_indices):
        component_indices = [component_indices]
    if len(component_indices) == 0:
        raise ValueError("component_indices must contain at least one component to drop.")

    indices = np.asarray(component_indices, dtype=int)
    if assume_one_indexed:
        indices = indices - 1
    if (indices < 0).any() or (indices >= n_components).any():
        raise ValueError("component_indices contain an unavailable component.")
    indices = np.unique(indices)
    keep = np.setdiff1d(np.arange(n_components), indices, assume_unique=False)
    if keep.size == 0:
        raise ValueError("Removing all components is not allowed.")

    if new_suffix is None:
        labels = "_".join(str(int(i + (1 if assume_one_indexed else 0))) for i in indices)
        new_suffix = f"no_components_{labels}"

    new_obsm_key = f"{base_obsm_key}_{new_suffix}"
    adata.obsm[new_obsm_key] = base_embed[:, keep]
    variance_ratio_key = None
    if pca_uns_key is not None and pca_uns_key in adata.uns:
        pca_entry = adata.uns[pca_uns_key]
        if isinstance(pca_entry, dict) and "variance_ratio" in pca_entry:
            variance_ratio_key = f"variance_ratio_{new_suffix}"
            pca_entry[variance_ratio_key] = np.asarray(pca_entry["variance_ratio"])[keep]

    varm_key = None
    if base_varm_key in adata.varm:
        varm_key = f"{base_varm_key}_{new_suffix}"
        adata.varm[varm_key] = np.asarray(adata.varm[base_varm_key])[:, keep]
    return {"obsm": new_obsm_key, "varm": varm_key, "variance_ratio": variance_ratio_key}
