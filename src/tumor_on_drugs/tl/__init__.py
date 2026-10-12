"""Analysis and annotation functions."""

from .annotate_clusters import annotate_clusters
from .check_gene_in_adata import check_gene_in_adata
from .umap_based_on_paga import umap_based_on_paga
from .network import (
    all_pairs,
    build_ego_graph,
    classify_gene,
    compute_condition_network,
    differential_network,
    edge_keys,
    ensure_outdir,
    fit_nb_marginals,
    get_counts_matrix,
    hvg_gene_list,
    load_table,
    overlap_test,
    per_gene_rewiring,
    plot_ego_graph,
    prepare_edge_table,
    select_genes,
    subset_window,
    summarize_categories,
    top_neighbors_for_gene,
)

__all__ = [
    "all_pairs",
    "annotate_clusters",
    "build_ego_graph",
    "check_gene_in_adata",
    "classify_gene",
    "compute_condition_network",
    "differential_network",
    "edge_keys",
    "ensure_outdir",
    "fit_nb_marginals",
    "get_counts_matrix",
    "hvg_gene_list",
    "load_table",
    "overlap_test",
    "per_gene_rewiring",
    "plot_ego_graph",
    "prepare_edge_table",
    "select_genes",
    "subset_window",
    "summarize_categories",
    "top_neighbors_for_gene",
    "umap_based_on_paga",
]
