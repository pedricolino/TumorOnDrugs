"""Preprocessing and data preparation functions."""

from .data import basic_checks, ensure_raw_counts_in_X, is_integerish_matrix
from .drop_components_from_embedding import drop_components_from_embedding
from .ica_single_run import ica_single_run
from .mstd import mstd, select_mstd_components, select_stable_components
from .redundancy import cluster_components, component_gene_redundancy
from .robustica import robustica
from .score_component_associations import compute_score_component_associations
from .scoring import compute_composite_score, safe_score_genes
from .stabilized_ica import stabilized_ica

__all__ = [
    "basic_checks",
    "cluster_components",
    "component_gene_redundancy",
    "compute_composite_score",
    "compute_score_component_associations",
    "drop_components_from_embedding",
    "ensure_raw_counts_in_X",
    "ica_single_run",
    "is_integerish_matrix",
    "mstd",
    "robustica",
    "safe_score_genes",
    "select_mstd_components",
    "select_stable_components",
    "stabilized_ica",
]
