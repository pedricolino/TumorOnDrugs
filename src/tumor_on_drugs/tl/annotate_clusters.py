"""Cluster annotation helpers."""


def annotate_clusters(
    adata,
    dotplot_data,
    cluster_col="clusters",
    tx_col="tx_status",
    tx_value="treated",
    rank_genes_groups_key="rank_genes_groups",
):
    treated = adata[adata.obs[tx_col] == tx_value]
    ranks = treated.obs.groupby(cluster_col).size().sort_values(ascending=False).index.tolist()
    new_col = cluster_col + "_annot"
    adata.obs[new_col] = adata.obs[cluster_col].astype(str)
    for cluster in adata.obs[cluster_col].cat.categories:
        try:
            label = dotplot_data.loc[:, cluster][
                dotplot_data.loc[:, cluster] == dotplot_data.loc[:, cluster].max()
            ].index[0].split(": ")[1]
        except Exception:
            label = ", ".join(adata.uns[rank_genes_groups_key]["names"][cluster][:5])
        adata.obs.loc[adata.obs[cluster_col] == cluster, new_col] = (
            f"{ranks.index(cluster)}: {label}"
        )
    adata.obs[new_col] = adata.obs[new_col].astype("category")
    categories = sorted(
        adata.obs[new_col].cat.categories,
        key=lambda value: int(value.split(": ")[0]),
    )
    adata.obs[new_col] = adata.obs[new_col].cat.set_categories(categories)
