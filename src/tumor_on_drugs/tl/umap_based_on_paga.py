"""PAGA-initialized UMAP layout."""


def umap_based_on_paga(
    adata,
    random_state=0,
    min_dist=0.3,
    spread=1.5,
    key_added=None,
    neighbors_key=None,
    resolution=1.0,
):
    import scanpy as sc
    import rapids_singlecell as rsc

    current_resolution = resolution
    rsc.tl.leiden(
        adata,
        random_state=random_state,
        resolution=current_resolution,
        key_added="leiden_for_reproducibility",
        n_iterations=400,
        neighbors_key=neighbors_key,
    )
    worked = False
    while not worked:
        try:
            sc.tl.paga(adata, groups="leiden_for_reproducibility")
            sc.pl.paga(
                adata,
                show=False,
                save="paga_used_for_umap_initiation.png",
                color="sample",
                random_state=random_state,
            )
            worked = True
        except Exception:
            if current_resolution > 1.2:
                print(
                    "Resolution exceeded 1.3, stopping retries.\n"
                    "PAGA failed, using random initialization for UMAP."
                )
                break
            current_resolution = round(current_resolution + 0.1, 1)
            print(f"PAGA failed, increasing resolution to {current_resolution} and retrying.")
            rsc.tl.leiden(
                adata,
                random_state=random_state,
                resolution=current_resolution,
                key_added="leiden_for_reproducibility",
                n_iterations=400,
                neighbors_key=neighbors_key,
            )
    rsc.tl.umap(
        adata,
        random_state=random_state,
        min_dist=min_dist,
        spread=spread,
        init_pos="paga" if worked else "random",
        neighbors_key=neighbors_key,
        key_added=key_added,
    )
