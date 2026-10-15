# Tumor On Drugs

Reusable Python helpers for single-cell tumor-on-drugs analyses, including
embedding plots, cluster composition pies, gene scoring, and enrichment plots.

## Install

```bash
conda run -n rsc_cuda13 python -m pip install -e ".[all,dev]"
```

The base package contains plotting dependencies. Install the `scanpy` extra
for AnnData and Scanpy-backed helpers, and the `gene-lookup` extra for gene
synonym lookup. GPU-backed helpers such as `compute_composite_score` expect
`rapids_singlecell` to be provided by the CUDA-enabled conda environment.

## Use

```python
import tumor_on_drugs as tod
from tumor_on_drugs import pp, pl, tl

result = pl.split_dim_plot(adata, split_by="treatment", color="leiden")
pl.add_pie_plot_overlay(adata, ax=ax, cluster_key="leiden", obs_key="phase")
pp.safe_score_genes(adata, gene_list=genes, score_name="signature")
tl.umap_based_on_paga(adata)
adata, removed = tod.ut.read_h5ad_dropping_incompatible("data.h5ad")
```

The package follows a Scanpy-style API: use `tumor_on_drugs.pp` for
preprocessing, `tumor_on_drugs.tl` for analysis, and `tumor_on_drugs.pl` for
plotting. `read_h5ad_dropping_incompatible` is available from `tumor_on_drugs.ut`
and requires the `scanpy` extra.

`pp.robustica(adata, n_comps=30, robust_runs=100)` runs RobustICA;
RobustICA estimator options such as `robust_method`, `robust_kws`, and
`random_state` can be passed as keyword arguments. It may return fewer
components than requested; the returned count is used for all stored arrays.
`pp.stabilized_ica(adata, ...)` runs stabilized ICA and requires the
`stabilized-ica` dependency. In the default gene space, cell scores and gene
loadings are stored in `adata.obsm["X_ica"]` and `adata.varm["ICs"]`,
respectively.
`pp.mstd(adata, ...)` stores H5AD-compatible stability summaries in
`adata.uns["ica"]["mstd"]` and returns `None`; use
`pl.plot_all_stability_indices(adata)` to inspect stability curves or
`pl.plot_mean_stability_index(adata)` to review their distributions. Its
plots show by default and close; pass `return_fig=True` to receive the
`(fig, ax)` pair, and `save=True` or a suffix to save under Scanpy's figure
directory. The stability-curve plot defaults to `optimal_n_compt="elbow"` and
selects the elbow automatically; pass `optimal_n_compt=None` to omit its marker.
`pp.select_mstd_components(adata)` can also select and store the elbow as
`optimal_n_compt`; `pp.stabilized_ica(adata)` uses that elbow automatically
when `n_comps` is omitted, while an explicit `n_comps` overrides it.
Stabilized ICA prints its
effective parameters by default; use `verbose=False` to silence that summary.
MSTD defaults use `algorithm="picard_ext"` and `n_jobs=-1`. Compute score
associations explicitly with
`pp.compute_score_component_associations(adata, score_key="...")`, then plot
the available metrics with `pl.plot_score_component_associations(adata, ...)`.
Associations are saved under `adata.uns["ica"]` by default, or
`adata.uns["pca"]` when `embedding_key` names a PCA embedding; `key_added` can
explicitly select a different storage key.
The plot saves its chosen `correlation_threshold` and optional
`mutual_info_threshold` alongside the stored association.
`pp.drop_components_from_embedding(adata, score_keys=[...])` uses those saved
cutoffs to create a reduced ICA embedding; alternatively, pass component
indices directly. MI filtering is disabled unless an MI cutoff is selected.
For example, after computing a score association, choose thresholds in the
plot and pass the score key to the drop helper:

```python
import matplotlib.pyplot as plt

pp.compute_score_component_associations(adata, "cell_cycle_score")
fig, axes = pl.plot_score_component_associations(
    adata, "cell_cycle_score", correlation_threshold=0.4
)
plt.show()
keys = pp.drop_components_from_embedding(
    adata, score_keys=["cell_cycle_score"]
)
```

`pp.mstd`, `pp.stabilized_ica`, and `pp.robustica` accept `use_pca=True` to use
`adata.obsm["X_pca"]` as input. In both ICA wrappers, PCA-space loadings are
projected back to genes with `adata.varm["PCs"]`. Input is centered by default
in all three functions; pass `center=False` to disable centering.
All three functions also accept `space="cell"` to run ICA with cells as features
and genes as observations; the default `space="gene"` keeps genes as features
and cells as observations. Cell-space ICA cannot be combined with
`use_pca=True`. The cell-aligned component values remain in `adata.obsm`,
while gene-aligned values are stored in `adata.varm`; these are cell weights
and gene scores, respectively, in cell space. `space` is recorded alongside
the ICA results. `pp.ica_single_run` accepts the same `space` option.

The former one-fit scikit-learn method is still available as
`pp.ica_single_run`.

MSTD displays a progress bar by default; pass `show_progress=False` to disable
it. The `stabilized-ica` extra installs the progress-bar dependency.

## Development

```bash
conda run -n rsc_cuda13 python -m pytest
conda run -n rsc_cuda13 python -m pip wheel . --no-deps -w dist
```