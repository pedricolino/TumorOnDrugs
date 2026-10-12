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

Stabilized ICA requires the `stabilized-ica` extra. The fitted scores and
loadings are stored in `adata.obsm["X_ica"]` and `adata.varm["ICs"]`.
`pp.mstd(adata, ...)` stores H5AD-compatible stability summaries in
`adata.uns["ica"]["mstd"]`; use `pl.plot_mean_stability_index(adata)` to review
them. Its defaults use `algorithm="picard_ext"` and `n_jobs=-1`. Compute score
associations explicitly with
`pp.compute_score_component_associations(adata, score_key="...")`, then plot
the available metrics with `pl.plot_score_component_associations(adata, ...)`.
The former one-fit scikit-learn method is still available as
`pp.ica_single_run`.

MSTD displays a progress bar by default; pass `show_progress=False` to disable
it. The `stabilized-ica` extra installs the progress-bar dependency.

## Development

```bash
conda run -n rsc_cuda13 python -m pytest
conda run -n rsc_cuda13 python -m pip wheel . --no-deps -w dist
```