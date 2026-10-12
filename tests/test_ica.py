import numpy as np
import pytest
from scipy import sparse


anndata = pytest.importorskip("anndata")

from tumor_on_drugs import pl, pp


def test_mstd_defaults_use_picard_ext_and_all_jobs():
    import inspect

    parameters = inspect.signature(pp.mstd).parameters
    assert parameters["algorithm"].default == "picard_ext"
    assert parameters["n_jobs"].default == -1
    assert parameters["show_progress"].default is True


def test_get_ica_matrix_converts_sparse_scalar_entries():
    from types import SimpleNamespace

    from tumor_on_drugs.pp._ica_utils import get_ica_matrix

    values = np.array(
        [
            [1.0, sparse.csr_matrix([[2.0]])],
            [3.0, sparse.csr_matrix([[4.0]])],
        ],
        dtype=object,
    )
    adata = SimpleNamespace(X=values, layers={}, var={}, n_vars=2)

    matrix, feature_mask = get_ica_matrix(adata, use_highly_variable=False)

    np.testing.assert_array_equal(feature_mask, [True, True])
    np.testing.assert_allclose(matrix, [[-1.0, -1.0], [1.0, 1.0]])


def test_get_ica_matrix_converts_gpu_sparse_input():
    from types import SimpleNamespace

    from tumor_on_drugs.pp._ica_utils import get_ica_matrix

    class FakeGpuSparse:
        format = "csr"
        shape = (2, 2)

        def __getitem__(self, key):
            raise AssertionError("GPU sparse matrices should be copied to CPU first.")

        def get(self):
            return sparse.csr_matrix([[1.0, 2.0], [3.0, 4.0]])

    adata = SimpleNamespace(X=FakeGpuSparse(), layers={}, var={}, n_vars=2)

    matrix, feature_mask = get_ica_matrix(adata, use_highly_variable=False)

    np.testing.assert_array_equal(feature_mask, [True, True])
    np.testing.assert_allclose(matrix, [[-1.0, -1.0], [1.0, 1.0]])


def test_get_ica_matrix_applies_highly_variable_mask_once():
    from tumor_on_drugs.pp._ica_utils import get_ica_matrix

    adata = anndata.AnnData(
        sparse.csr_matrix(
            [[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0], [5.0, 6.0, 7.0, 8.0]]
        )
    )
    adata.var["highly_variable"] = [True, False, True, False]

    matrix, feature_mask = get_ica_matrix(adata)

    np.testing.assert_array_equal(feature_mask, [True, False, True, False])
    np.testing.assert_allclose(matrix, [[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]])


def make_adata(n_obs=60, n_vars=12, seed=0):
    rng = np.random.default_rng(seed)
    sources = rng.laplace(size=(n_obs, min(4, n_vars)))
    mixing = rng.normal(size=(min(4, n_vars), n_vars))
    matrix = sources @ mixing + rng.normal(scale=0.05, size=(n_obs, n_vars))
    adata = anndata.AnnData(matrix)
    adata.var_names = [f"gene_{i}" for i in range(n_vars)]
    adata.var["highly_variable"] = True
    return adata


def test_public_ica_definitions_are_in_python_modules():
    import inspect

    assert inspect.getsourcefile(pp.ica).endswith("/pp/ica.py")
    assert inspect.getsourcefile(pp.ica_single_run).endswith("/pp/ica_single_run.py")
    assert inspect.getsourcefile(pl.plot_score_component_associations).endswith(
        "/pl/plot_score_component_associations.py"
    )


def test_fastica_single_run_writes_h5ad_compatible_metadata(tmp_path):
    adata = make_adata()

    pp.ica_single_run(
        adata,
        n_comps=3,
        n_top_genes=4,
        random_state=3,
        max_iter=200,
    )

    assert adata.obsm["X_ica"].shape == (adata.n_obs, 3)
    assert adata.varm["ica_components"].shape == (adata.n_vars, 3)
    assert adata.uns["ica"]["params"]["method"] == "fastica_single_run"
    assert "model" not in adata.uns["ica"]
    output = tmp_path / "fastica.h5ad"
    adata.write_h5ad(output, compression="gzip")
    assert anndata.read_h5ad(output).obsm["X_ica"].shape == (adata.n_obs, 3)


def test_stabilized_ica_and_mstd_store_h5ad_compatible_results(tmp_path):
    pytest.importorskip("sica")
    adata = make_adata()

    runs = pp.mstd(
        adata,
        m=2,
        M=4,
        step=2,
        n_runs=2,
        algorithm="fastica_par",
        max_iter=100,
    )
    assert set(runs) == {2, 4}
    assert adata.uns["ica"]["mstd"]["component_counts"].tolist() == [2, 4]
    selected = pp.select_mstd_components(adata, last_n=2)
    assert selected is None or selected in {2, 4}

    pp.ica(
        adata,
        n_comps=3,
        n_runs=2,
        algorithm="fastica_par",
        max_iter=100,
        n_top_genes=4,
    )
    assert adata.obsm["X_ica"].shape == (adata.n_obs, 3)
    assert adata.varm["ICs"].shape == (adata.n_vars, 3)
    assert len(adata.uns["ica"]["stability_indexes"]) == 3
    assert pp.select_stable_components(adata, min_stability=0).tolist() == [1, 2, 3]

    fig, _ = pl.plot_mean_stability_index(adata)
    import matplotlib.pyplot as plt

    plt.close(fig)
    output = tmp_path / "stabilized_ica.h5ad"
    adata.write_h5ad(output, compression="gzip")
    reopened = anndata.read_h5ad(output)
    assert reopened.uns["ica"]["mstd"]["stability_indexes"].shape == (2, 4)
    assert reopened.obsm["X_ica"].shape == (adata.n_obs, 3)


def test_score_associations_plot_available_metrics_and_roundtrip(tmp_path):
    adata = make_adata(n_obs=80, n_vars=10, seed=11)
    rng = np.random.default_rng(12)
    embedding = rng.normal(size=(adata.n_obs, 2))
    adata.obsm["X_ica"] = embedding
    adata.obs["score"] = 2 * embedding[:, 0] + rng.normal(scale=0.1, size=adata.n_obs)

    associations = pp.compute_score_component_associations(
        adata, "score", embedding_key="X_ica", method="spearman"
    )
    assert {"correlation", "abs_correlation", "p_value", "mutual_info"} <= set(
        associations.columns
    )
    fig, axes = pl.plot_score_component_associations(adata, "score")
    assert len(axes) == 2
    import matplotlib.pyplot as plt

    plt.close(fig)

    del adata.uns["ica"]["associations"]["score"]["mutual_info"]
    fig, axes = pl.plot_score_component_associations(adata, "score")
    assert len(axes) == 1
    plt.close(fig)

    output = tmp_path / "associations.h5ad"
    adata.write_h5ad(output, compression="gzip")
    reopened = anndata.read_h5ad(output)
    assert "correlation" in reopened.uns["ica"]["associations"]["score"]


def test_component_clustering_and_gene_redundancy():
    adata = make_adata(n_obs=50, n_vars=8, seed=17)
    base = np.linspace(-2, 2, adata.n_obs)
    adata.obsm["X_ica"] = np.column_stack(
        [base, -base + np.sin(base) * 0.01, np.random.default_rng(1).normal(size=adata.n_obs)]
    )
    loadings = np.zeros((adata.n_vars, 3))
    loadings[:4, 0] = [5, 4, 3, 2]
    loadings[:4, 1] = [4, 3, 2, 1]
    loadings[4:, 2] = [5, 4, 3, 2]
    adata.varm["ICs"] = loadings

    clusters = pp.cluster_components(adata, correlation_threshold=0.8)
    assert clusters["IC1"] == clusters["IC2"]
    assert clusters["IC1"] != clusters["IC3"]
    two_component_adata = make_adata(n_obs=50, n_vars=8, seed=18)
    two_component_adata.obsm["X_ica"] = adata.obsm["X_ica"][:, :2]
    assert len(pp.cluster_components(two_component_adata)) == 2

    redundancy = pp.component_gene_redundancy(adata, top_n=4)
    shared = redundancy.query("component_a == 'IC1' and component_b == 'IC2'").iloc[0]
    assert shared["n_shared"] == 4
    assert shared["jaccard"] == 1
