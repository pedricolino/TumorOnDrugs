def plot_enrichment_dotplot(
        adata=None, marker_genes=None,
        gprofiler_kwargs={'sources': ['GO:MF', 'GO:BP', 'KEGG', 'REAC']},
        top_n_genes=50, top_n_terms=10,
        max_log_pval=10, figsize_factor=(0.3, 0.15),
        fontsize=6, title="Enriched Terms per Cluster",
        group_type='Cluster', ax=None, incl_source_db=True,
        up_or_down='both', rank_genes_groups_key='rank_genes_groups',
        save=None,  # Scanpy-style: False | True | str suffix
        show=True
        ):
    """
    Create a dotplot showing enriched terms for each module/cluster.

    Parameters
    ----------
    up_or_down : {'up', 'down', 'both'}
        Default is 'both'.
    save : bool or str
        Scanpy-style save argument:
        - False: do not save
        - True: save to default filename in `sc.settings.figdir`
        - str: suffix appended to default filename in `sc.settings.figdir`
    show : bool
        Whether to show the plot via plt.show().
    """
    import os
    from pathlib import Path
    import re
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import scanpy as sc

    valid_directions = {'up', 'down', 'both'}
    if up_or_down not in valid_directions:
        raise ValueError(f"`up_or_down` must be one of {valid_directions}, got '{up_or_down}'.")

    if gprofiler_kwargs is None:
        gprofiler_kwargs = {}

    if marker_genes is None:
        if adata is None:
            raise ValueError("Provide either `marker_genes` or `adata`.")
        if rank_genes_groups_key not in adata.uns:
            raise KeyError(f"`{rank_genes_groups_key}` not found in adata.uns.")

        obs_group = adata.uns[rank_genes_groups_key]['params']['groupby']

        marker_genes = {}
        for group in adata.obs[obs_group].unique().categories:
            cluster_genes = adata.uns[rank_genes_groups_key]['names'][group]
            lfc = adata.uns[rank_genes_groups_key]['logfoldchanges'][group]

            # find idx that nan
            nan_idx = np.where(cluster_genes.astype(str)==np.str_(np.nan))
            # remove nan from cluster_genes and lfc
            cluster_genes = np.delete(cluster_genes, nan_idx)
            lfc = np.delete(lfc, nan_idx)

            if up_or_down == 'up':
                cluster_genes = cluster_genes[lfc > 0]
            elif up_or_down == 'down':
                cluster_genes = cluster_genes[lfc < 0]
            # both -> no filtering

            marker_genes[group] = cluster_genes[:top_n_genes]

    enrichments = {}
    for m in marker_genes.keys():
        genes = marker_genes[m]
        gene_list = genes.tolist() if hasattr(genes, "tolist") else list(genes)
        if len(gene_list) == 0:
            continue
        enrich = sc.queries.enrich(
            gene_list,
            org="hsapiens",
            gprofiler_kwargs=gprofiler_kwargs
        )
        if enrich is not None and len(enrich) > 0:
            enrichments[m] = enrich

    top_terms = []
    for i in enrichments:
        top = enrichments[i].head(top_n_terms).copy()
        top[group_type] = i
        top_terms.append(top)

    if not top_terms:
        print("No enrichment results found.")
        return None

    top_terms = pd.concat(top_terms, ignore_index=True)

    if incl_source_db:
        top_terms['source_name'] = top_terms['source'] + ': ' + top_terms['name']
        dotplot_data = top_terms.pivot(index='source_name', columns=group_type, values='p_value')
    else:
        dotplot_data = top_terms.pivot(index='name', columns=group_type, values='p_value')

    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(r'([0-9]+)', str(s))]

    try:
        numeric_cols = pd.to_numeric(dotplot_data.columns)
        sorted_cols = dotplot_data.columns[np.argsort(numeric_cols)]
    except (ValueError, TypeError):
        sorted_cols = sorted(dotplot_data.columns, key=natural_sort_key)

    dotplot_data = dotplot_data[sorted_cols]

    dotplot_data[group_type] = dotplot_data.idxmin(axis=1)
    try:
        dotplot_data[group_type] = pd.to_numeric(dotplot_data[group_type])
        dotplot_data = dotplot_data.sort_values(by=group_type)
    except (ValueError, TypeError):
        dotplot_data = dotplot_data.sort_values(by=group_type, key=lambda x: x.map(natural_sort_key))

    dotplot_data = dotplot_data.drop(columns=[group_type])
    dotplot_data = dotplot_data.fillna(1)
    dotplot_data = -np.log10(dotplot_data)
    dotplot_data = dotplot_data.clip(upper=max_log_pval)

    fig_width = figsize_factor[0] * dotplot_data.shape[1]
    fig_height = figsize_factor[1] * dotplot_data.shape[0]

    if ax is None:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    else:
        fig = ax.figure

    sns.heatmap(
        dotplot_data,
        annot=False,
        cmap='viridis',
        ax=ax,
        cbar_kws={'label': '-log10(p-value)'}
    )

    ax.set_yticks(np.arange(dotplot_data.shape[0]) + 0.5)
    ax.set_yticklabels(dotplot_data.index, rotation=0, fontsize=fontsize)
    ax.set_xticks(np.arange(dotplot_data.shape[1]) + 0.5)
    ax.set_xticklabels(dotplot_data.columns, rotation=45, ha='right', fontsize=fontsize)
    ax.set_title(title)
    ax.set_xlabel(group_type)
    ax.grid(False)

    if up_or_down == 'up':
        ax.set_ylabel('Enriched terms (upregulated genes)')
    elif up_or_down == 'down':
        ax.set_ylabel('Enriched terms (downregulated genes)')
    else:
        ax.set_ylabel('Enriched terms (all marker genes)')

    plt.tight_layout()

    if save is not None:
        path = sc.settings.figdir / save
        plt.savefig(path, bbox_inches='tight')

    if show:
        plt.show()

    return dotplot_data

