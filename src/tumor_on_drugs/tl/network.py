"""Gene-network analysis helpers used by cluster characterization workflows."""

import json
import os
from itertools import combinations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse
from scipy.stats import chi2, fisher_exact


def get_counts_matrix(adata, raw_layer="raw"):
    """Return an AnnData expression matrix as dense integer-like counts."""
    X = adata.layers[raw_layer] if raw_layer is not None and raw_layer in adata.layers else adata.X
    if sparse.issparse(X):
        X = X.toarray()
    return np.asarray(X, dtype=np.int32)


def hvg_gene_list(adata, n_top=2000):
    """Select highly variable genes with Scanpy fallbacks."""
    ad = adata.copy()
    for flavor in ("seurat_v3", "cell_ranger"):
        try:
            sc.pp.highly_variable_genes(ad, n_top_genes=n_top, flavor=flavor)
            genes = ad.var_names[ad.var["highly_variable"].to_numpy(copy=False)].to_numpy()
            if genes.size:
                return genes
        except Exception:
            pass
    X = ad.X.toarray() if sparse.issparse(ad.X) else np.asarray(ad.X)
    return ad.var_names.to_numpy()[np.argsort(-X.mean(axis=0))[:n_top]]


def select_genes(adata, hvg_n_top=2000, network_n_genes=300):
    return np.asarray(hvg_gene_list(adata, n_top=hvg_n_top)[:network_n_genes], dtype=str)


def fit_nb_marginals(counts_dense, eps=1e-12, r_max=1e6):
    X = np.asarray(counts_dense, dtype=np.float64)
    mu = np.maximum(X.mean(axis=0), eps)
    var = X.var(axis=0, ddof=1) if X.shape[0] > 1 else np.zeros_like(mu)
    denom = var - mu
    r = np.full_like(mu, float(r_max))
    mask = denom > 0
    r[mask] = mu[mask] ** 2 / denom[mask]
    return mu, np.clip(r, 1e-8, float(r_max))


def all_pairs(n):
    pairs = np.asarray(list(combinations(range(n), 2)), dtype=np.int32)
    return pairs[:, 0], pairs[:, 1]


def edge_keys(genes, pairs_i, pairs_j):
    first, second = np.asarray(genes[pairs_i], dtype=object), np.asarray(genes[pairs_j], dtype=object)
    swap = first > second
    return np.where(swap, second, first), np.where(swap, first, second)


def compute_condition_network(
    adata,
    condition_name,
    out_prefix,
    *,
    raw_layer=None,
    hvg_n_top=2000,
    network_n_genes=300,
    max_cells=20000,
    min_pop=25,
    min_expected=5.0,
    pval_threshold=0.05,
    seed=0,
):
    """Infer one MIMIQ network and write its edge table and metadata."""
    from mimiq import mi_negative_binomial_crossfit

    rng = np.random.default_rng(seed)
    if adata.n_obs == 0:
        raise ValueError(f"{condition_name}: no cells in subset")
    ad = adata
    if ad.n_obs > max_cells:
        ad = ad[rng.choice(ad.n_obs, size=max_cells, replace=False)].copy()
    genes_use = select_genes(ad, hvg_n_top=hvg_n_top, network_n_genes=network_n_genes)
    ad = ad[:, genes_use].copy()
    counts = get_counts_matrix(ad, raw_layer=raw_layer)
    genes = ad.var_names.to_numpy(dtype=str)
    pairs_i, pairs_j = all_pairs(len(genes))
    gene1, gene2 = edge_keys(genes, pairs_i, pairs_j)
    mu, r = fit_nb_marginals(counts)
    mi_mat, chi2_cv, dof_cv = mi_negative_binomial_crossfit(
        counts, mu, r, min_pop=int(min_pop), seed=int(seed),
        min_expected=float(min_expected), use_empirical_marginals=False,
    )
    mi_edges = np.asarray(mi_mat)[pairs_i, pairs_j]
    chi2_edges = np.asarray(chi2_cv)[pairs_i, pairs_j]
    dof_edges = np.asarray(dof_cv)[pairs_i, pairs_j]
    pval = np.full(mi_edges.shape, np.nan, dtype=float)
    valid = np.isfinite(chi2_edges) & np.isfinite(dof_edges) & (dof_edges > 0) & (chi2_edges >= 0)
    pval[valid] = chi2.sf(chi2_edges[valid], dof_edges[valid])
    significant = np.isfinite(mi_edges) & np.isfinite(pval) & (pval <= pval_threshold)
    edge_df = pd.DataFrame({
        "gene1": gene1, "gene2": gene2, "mi_raw": mi_edges,
        "chi2_cv": chi2_edges, "dof_cv": dof_edges, "pval": pval,
        "significant": significant, "mi_sig": np.where(significant, mi_edges, 0.0),
    })
    edge_df.to_csv(f"{out_prefix}_edges.csv.gz", index=False, compression="gzip")
    with open(f"{out_prefix}_meta.json", "w") as handle:
        json.dump({"condition": condition_name, "n_cells": int(ad.n_obs), "n_genes": int(ad.n_vars), "genes_selected": genes.tolist()}, handle, indent=2)
    return edge_df


def differential_network(df0, df1, label0="cond0", label1="cond1"):
    merged = df0.merge(df1, on=["gene1", "gene2"], suffixes=(f"_{label0}", f"_{label1}"), how="inner")
    merged["delta_mi"] = merged[f"mi_sig_{label1}"] - merged[f"mi_sig_{label0}"]
    merged["abs_delta_mi"] = merged["delta_mi"].abs()
    merged["sig_both"] = merged[f"significant_{label0}"].astype(bool) & merged[f"significant_{label1}"].astype(bool)
    merged["sig_either"] = merged[f"significant_{label0}"].astype(bool) | merged[f"significant_{label1}"].astype(bool)
    return merged


def per_gene_rewiring(dmin, use_sig_only=True):
    data = dmin[dmin["sig_both"]].copy() if use_sig_only else dmin.copy()
    genes = pd.concat([data[["gene1", "abs_delta_mi"]].rename(columns={"gene1": "gene"}), data[["gene2", "abs_delta_mi"]].rename(columns={"gene2": "gene"})])
    if genes.empty:
        return pd.DataFrame({"gene": [], "rewiring_score": []})
    return genes.groupby("gene", as_index=False)["abs_delta_mi"].sum().rename(columns={"abs_delta_mi": "rewiring_score"}).sort_values("rewiring_score", ascending=False).reset_index(drop=True)


def subset_window(adata, time_key, arm_key, time_values, arm_value, cluster_key=None, cluster_value=None):
    mask = adata.obs[time_key].astype(str).isin([str(value) for value in time_values]).to_numpy()
    mask &= adata.obs[arm_key].astype(str).eq(str(arm_value)).to_numpy()
    if cluster_key is not None and cluster_value is not None:
        mask &= adata.obs[cluster_key].astype(str).eq(str(cluster_value)).to_numpy()
    return adata[mask].copy()


def classify_gene(gene):
    gene = str(gene)
    categories = {
        "cell_cycle": {"CENPF", "NUF2", "LBR", "RRM2", "CKAP2L", "NEK2", "CDCA7", "CDCA8", "BUB1", "STIL", "CKS1B", "MCM6", "ORC1", "CLSPN", "GEN1", "EXO1", "RAD54L", "CDC20", "CENPA", "CENPL", "PSRC1", "KIF2C", "KIF14", "SPC25", "E2F2"},
        "stress_response": {"TXNIP", "ATF3", "BTG2", "DUSP10", "GADD45A", "IER5", "SESN2", "NR4A2", "HSPA6", "RGS16", "ERRFI1", "ISG15", "IFI6", "IFI44", "RSAD2", "IFIT1", "IFIT2", "IFIT3"},
        "immune_inflammation": {"IL1B", "IL1RN", "IL1R2", "CTSS", "CTSE", "SLAMF6", "CD84", "THEMIS2", "INAVA", "C4BPB", "IFI6", "IFI44", "RSAD2", "ISG15"},
        "epithelial_adhesion": {"FABP1", "LAMB3", "LAMC2", "MUC1", "TACSTD2", "NECTIN4", "EPHA2", "EPHX1", "PTPRU", "LGR6", "PROX1", "GUCA2A", "GUCA2B", "REG4", "CLCA4", "PIGR", "GPA33", "KCNK1", "VANGL2", "LBH", "CGA", "CA6", "SLC3A1"},
        "metabolism_redox": {"SELENBP1", "SLC40A1", "GCLM", "SLC1A4", "SLC16A1", "HMGCS2", "PCSK9", "CYBRD1", "ACOT7", "CTH", "CHI3L1", "KHK", "SLC20A1", "SLC2A1", "SRM", "ODC1", "CTPS1", "GALE", "AMPD1", "DHRS9", "CDA", "SERINC2", "TLCD4"},
        "dna_repair_replication": {"MSH2", "MSH6", "USP1", "RRM2", "RAD54L", "EXO1", "GEN1", "ORC1", "CLSPN", "NABP1", "E2F2"},
    }
    return next((name for name, genes in categories.items() if gene in genes), "other")


def summarize_categories(data, gene_col="gene"):
    result = data.copy()
    result["category"] = result[gene_col].map(classify_gene)
    return result.groupby("category", as_index=False).agg(n_genes=(gene_col, "count"), mean_rewiring=("rewiring_score_late", "mean"), median_rewiring=("rewiring_score_late", "median")).sort_values("n_genes", ascending=False)


def overlap_test(top_genes, cc_genes):
    top, cc = set(map(str, top_genes)), set(map(str, cc_genes))
    universe = top | cc
    a, b, c = len(top & cc), len(top - cc), len(cc - top)
    table = [[a, b], [c, max(1, len(universe) - a - b - c)]]
    oddsratio, pvalue = fisher_exact(table, alternative="greater")
    return {"overlap": a, "top_n": len(top), "cc_n": len(cc), "oddsratio": oddsratio, "pvalue": pvalue, "table": table, "overlap_genes": sorted(top & cc)}


def prepare_edge_table(path, label):
    data = pd.read_csv(path).copy()
    required = {"gene1", "gene2", "mi_raw", "pval", "significant", "mi_sig"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"{path} missing columns: {sorted(missing)}")
    data["label"] = label
    return data


def load_table(path):
    """Load a CSV table and raise a clear error when it is absent."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def ensure_outdir(path):
    """Create an output directory and return its path."""
    os.makedirs(path, exist_ok=True)
    return path


def top_neighbors_for_gene(dmin, gene, top_n=20):
    gene = str(gene)
    data = dmin[(dmin["gene1"].astype(str) == gene) | (dmin["gene2"].astype(str) == gene)].copy()
    if data.empty:
        return data
    data["neighbor"] = np.where(data["gene1"].astype(str) == gene, data["gene2"], data["gene1"])
    return data.sort_values("abs_delta_mi", ascending=False).head(top_n)[["gene1", "gene2", "neighbor", "mi_sig_pre", "mi_sig_late", "delta_mi", "abs_delta_mi", "significant_pre", "significant_late"]]


def build_ego_graph(dmin, focal_gene, top_neighbors=20):
    edges = top_neighbors_for_gene(dmin, focal_gene, top_neighbors)
    if edges.empty:
        return None, None
    graph = nx.Graph()
    graph.add_node(str(focal_gene))
    for row in edges.itertuples(index=False):
        graph.add_edge(str(row.gene1), str(row.gene2), weight=float(row.abs_delta_mi), delta_mi=float(row.delta_mi))
    return graph, edges


def plot_ego_graph(graph, focal_gene, out_png, title=""):
    if graph is None or graph.number_of_nodes() == 0:
        return
    pos = nx.spring_layout(graph, seed=0, weight="weight")
    plt.figure(figsize=(8, 6))
    weights = nx.get_edge_attributes(graph, "weight")
    max_weight = max(weights.values(), default=1.0)
    nx.draw_networkx_edges(graph, pos, width=[0.5 + 4 * weights[edge] / max_weight for edge in graph.edges()], alpha=0.7)
    nx.draw_networkx_nodes(graph, pos, node_color=["#FF715B" if node == focal_gene else "#CFCFCF" for node in graph], node_size=[700 if node == focal_gene else 300 for node in graph])
    nx.draw_networkx_labels(graph, pos, font_size=10)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()
