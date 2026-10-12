"""Gene-set scoring helpers."""

from .data import is_integerish_matrix
from ..ut.environment import check_prepare_gpu


def safe_score_genes(adata, gene_list, score_name: str, use_raw: bool = False, use_gpu: bool = None, normalize_if_necessary: bool = False):
    import scanpy as sc

    if use_gpu is None:
        use_gpu = check_prepare_gpu(verbose=False)
    matrix = adata.raw.X if use_raw and adata.raw is not None else adata.X
    if is_integerish_matrix(matrix):
        print("Data to use is not log-normalized.")
        if not normalize_if_necessary:
            raise ValueError("Data to use is not log-normalized. Set normalize_if_necessary=True or normalize manually before scoring.")
        if use_gpu:
            import rapids_singlecell as rsc
            rsc.get.anndata_to_GPU(adata)
            rsc.pp.normalize_total(adata, target_sum=1e4)
            rsc.pp.log1p(adata)
            rsc.get.anndata_to_CPU(adata)
        else:
            sc.pp.normalize_total(adata, target_sum=1e4)
            sc.pp.log1p(adata)

    genes = [gene for gene in gene_list if gene in adata.var_names]
    if len(genes) < 5:
        print(f"[score_genes] Skip {score_name}: only {len(genes)} genes found in var_names.")
        return
    scorer = None
    if use_gpu:
        import rapids_singlecell as rsc
        scorer = rsc.tl.score_genes
    else:
        scorer = sc.tl.score_genes
    try:
        scorer(adata, gene_list=genes, score_name=score_name, use_raw=use_raw)
        print(f"Added {score_name} using {len(genes)} out of {len(gene_list)} provided genes.")
    except Exception as exc:
        print(f"[score_genes] Failed {score_name}: {exc!r}")


def compute_composite_score(adata, up_genes, down_genes, score_name: str, use_raw=False, save_intermediate_scores=False):
    import rapids_singlecell as rsc

    rsc.tl.score_genes(adata, gene_list=up_genes, score_name=score_name + "_up", use_raw=use_raw)
    rsc.tl.score_genes(adata, gene_list=down_genes, score_name=score_name + "_down", use_raw=use_raw)
    adata.obs[score_name] = adata.obs[score_name + "_up"] - adata.obs[score_name + "_down"]
    if save_intermediate_scores:
        del adata.obs[score_name + "_up"]
        del adata.obs[score_name + "_down"]
