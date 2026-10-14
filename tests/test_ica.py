import sys

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
    assert parameters["use_pca"].default is False
    assert parameters["center"].default is True
    ica_parameters = inspect.signature(pp.ica).parameters
    assert ica_parameters["n_comps"].default is None
    assert ica_parameters["verbose"].default is True
    assert ica_parameters["use_pca"].default is False
    assert ica_parameters["center"].default is True


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
    adata = SimpleNamespace(X=values, layers={}, var={}, n_obs=2, n_vars=2)

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

    adata = SimpleNamespace(
        X=FakeGpuSparse(), layers={}, var={}, n_obs=2, n_vars=2
    )

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


def test_get_ica_matrix_uses_pca_scores_and_optional_centering():
    from types import SimpleNamespace

    from tumor_on_drugs.pp._ica_utils import get_ica_matrix

    scores = np.array([[1.0, 2.0], [3.0, 5.0], [6.0, 8.0]])
    adata = SimpleNamespace(
        obsm={"X_pca": scores},
        var={},
        n_obs=3,
        n_vars=4,
    )

    centered, feature_mask = get_ica_matrix(adata, use_pca=True)
    uncentered, _ = get_ica_matrix(adata, use_pca=True, center=False)

    np.testing.assert_array_equal(feature_mask, np.ones(4, dtype=bool))
    np.testing.assert_allclose(centered.mean(axis=0), [0.0, 0.0], atol=1e-15)
    np.testing.assert_array_equal(uncentered, scores)


def test_ica_pca_input_projects_loadings_back_to_genes(monkeypatch):
    from types import ModuleType

    adata = anndata.AnnData(np.zeros((4, 3)))
    adata.var_names = ["gene_a", "gene_b", "gene_c"]
    scores = np.array([[1.0, 2.0], [3.0, 5.0], [6.0, 8.0], [10.0, 11.0]])
    pca_components = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    adata.obsm["X_pca"] = scores
    adata.varm["PCs"] = pca_components
    fitted_matrices = []
    transformed_matrices = []

    class FakeStabilizedICA:
        def __init__(self, **kwargs):
            pass

        def fit(self, matrix):
            fitted_matrices.append(matrix.copy())
            self.S_ = np.array(
                [[1.0, 3.0, 6.0, 10.0], [2.0, 5.0, 8.0, 11.0]]
            )
            self.stability_indexes_ = np.array([0.8, 0.9])

        def transform(self, matrix):
            transformed_matrices.append(matrix.copy())
            return np.array([[2.0, 0.0], [0.0, 3.0]])

    sica_module = ModuleType("sica")
    base_module = ModuleType("sica.base")
    base_module.StabilizedICA = FakeStabilizedICA
    sica_module.base = base_module
    monkeypatch.setitem(sys.modules, "sica", sica_module)
    monkeypatch.setitem(sys.modules, "sica.base", base_module)

    pp.ica(
        adata,
        n_comps=2,
        use_pca=True,
        center=False,
        n_top_genes=2,
        verbose=False,
    )

    np.testing.assert_array_equal(fitted_matrices[0], scores.T)
    np.testing.assert_array_equal(transformed_matrices[0], scores.T)
    np.testing.assert_array_equal(
        adata.obsm["X_ica"],
        np.array([[1.0, 2.0], [3.0, 5.0], [6.0, 8.0], [10.0, 11.0]]),
    )
    np.testing.assert_allclose(
        adata.varm["ICs"],
        pca_components @ np.array([[2.0, 0.0], [0.0, 3.0]]),
    )
    assert adata.uns["ica"]["params"]["use_pca"] is True
    assert adata.uns["ica"]["params"]["center"] is False


def test_ica_cell_space_stores_cell_and_gene_values_on_matching_axes(monkeypatch):
    from types import ModuleType

    adata = anndata.AnnData(np.arange(12, dtype=float).reshape(4, 3))
    adata.var_names = ["gene_a", "gene_b", "gene_c"]
    fitted_matrices = []
    transformed_matrices = []

    class FakeStabilizedICA:
        def __init__(self, **kwargs):
            pass

        def fit(self, matrix):
            fitted_matrices.append(matrix.copy())
            self.S_ = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
            self.stability_indexes_ = np.array([0.8, 0.9])

        def transform(self, matrix):
            transformed_matrices.append(matrix.copy())
            return np.arange(8, dtype=float).reshape(2, 4)

    sica_module = ModuleType("sica")
    base_module = ModuleType("sica.base")
    base_module.StabilizedICA = FakeStabilizedICA
    sica_module.base = base_module
    monkeypatch.setitem(sys.modules, "sica", sica_module)
    monkeypatch.setitem(sys.modules, "sica.base", base_module)

    pp.ica(
        adata,
        n_comps=2,
        use_highly_variable=False,
        space="cell",
        n_top_genes=2,
        verbose=False,
    )

    np.testing.assert_array_equal(fitted_matrices[0], adata.X)
    np.testing.assert_array_equal(transformed_matrices[0], adata.X)
    np.testing.assert_array_equal(
        adata.obsm["X_ica"], np.arange(8, dtype=float).reshape(4, 2)
    )
    np.testing.assert_array_equal(
        adata.varm["ICs"], np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
    )
    assert adata.uns["ica"]["params"]["space"] == "cell"
    assert adata.uns["ica"]["top_genes_positive"][0].tolist() == [
        "gene_c",
        "gene_b",
    ]


def test_ica_without_estimator_whitening_prewhitens_to_component_count(monkeypatch):
    from types import ModuleType

    adata = anndata.AnnData(np.arange(18, dtype=float).reshape(6, 3))
    adata.var_names = ["gene_a", "gene_b", "gene_c"]
    whitening_inputs = []
    whitening_options = []
    fitted_matrices = []
    transformed_matrices = []

    def whitening(matrix, *, n_components, zero_center, **kwargs):
        whitening_inputs.append(matrix.copy())
        whitening_options.append((n_components, zero_center))
        return np.arange(12, dtype=float).reshape(6, 2), None

    class FakeStabilizedICA:
        def __init__(self, *, whiten, **kwargs):
            assert whiten is False

        def fit(self, matrix):
            fitted_matrices.append(matrix.copy())
            self.S_ = np.arange(12, dtype=float).reshape(2, 6)
            self.stability_indexes_ = np.array([0.8, 0.9])

        def transform(self, matrix):
            transformed_matrices.append(matrix.copy())
            return np.ones((3, 2))

    sica_module = ModuleType("sica")
    base_module = ModuleType("sica.base")
    base_module.whitening = whitening
    base_module.StabilizedICA = FakeStabilizedICA
    sica_module.base = base_module
    monkeypatch.setitem(sys.modules, "sica", sica_module)
    monkeypatch.setitem(sys.modules, "sica.base", base_module)

    pp.ica(
        adata,
        n_comps=2,
        use_highly_variable=False,
        center=False,
        whiten=False,
        n_top_genes=2,
        verbose=False,
    )

    np.testing.assert_array_equal(whitening_inputs[0], adata.X)
    assert whitening_options == [(2, False)]
    np.testing.assert_array_equal(fitted_matrices[0], np.arange(12).reshape(6, 2).T)
    np.testing.assert_array_equal(transformed_matrices[0], adata.X.T)
    assert adata.obsm["X_ica"].shape == (adata.n_obs, 2)
    assert adata.varm["ICs"].shape == (adata.n_vars, 2)


def test_mstd_pca_input_respects_center_option(monkeypatch):
    from types import ModuleType

    adata = anndata.AnnData(np.zeros((5, 3)))
    scores = np.array(
        [[1.0, 2.0], [3.0, 5.0], [6.0, 8.0], [10.0, 11.0], [12.0, 14.0]]
    )
    adata.obsm["X_pca"] = scores
    whitening_inputs = []
    zero_center_values = []

    def whitening(matrix, *, zero_center, **kwargs):
        whitening_inputs.append(matrix.copy())
        zero_center_values.append(zero_center)
        return matrix, None

    class FakeStabilizedICA:
        def __init__(self, n_components, **kwargs):
            self.n_components = n_components

        def fit(self, matrix):
            self.stability_indexes_ = np.full(self.n_components, 0.75)

    sica_module = ModuleType("sica")
    base_module = ModuleType("sica.base")
    base_module.whitening = whitening
    base_module.StabilizedICA = FakeStabilizedICA
    sica_module.base = base_module
    monkeypatch.setitem(sys.modules, "sica", sica_module)
    monkeypatch.setitem(sys.modules, "sica.base", base_module)

    pp.mstd(
        adata,
        m=2,
        M=2,
        n_runs=1,
        use_pca=True,
        center=False,
        show_progress=False,
    )

    np.testing.assert_array_equal(whitening_inputs[0], scores.T)
    assert zero_center_values == [False]
    assert adata.uns["ica"]["mstd"]["params"]["use_pca"] is True
    assert adata.uns["ica"]["mstd"]["params"]["center"] is False


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


def test_fastica_single_run_cell_space_stores_cell_and_gene_values():
    adata = make_adata(n_obs=20, n_vars=8)

    pp.ica_single_run(
        adata,
        n_comps=3,
        space="cell",
        random_state=3,
        max_iter=200,
    )

    assert adata.obsm["X_ica"].shape == (adata.n_obs, 3)
    assert adata.varm["ica_components"].shape == (adata.n_vars, 3)
    assert adata.uns["ica"]["params"]["space"] == "cell"


def test_mstd_cell_space_uses_genes_as_observations(monkeypatch):
    from types import ModuleType

    adata = anndata.AnnData(np.zeros((4, 3)))
    whitening_inputs = []
    fitted_matrices = []

    def whitening(matrix, **kwargs):
        whitening_inputs.append(matrix.copy())
        return matrix, None

    class FakeStabilizedICA:
        def __init__(self, n_components, **kwargs):
            self.n_components = n_components

        def fit(self, matrix):
            fitted_matrices.append(matrix.copy())
            self.stability_indexes_ = np.full(self.n_components, 0.75)

    sica_module = ModuleType("sica")
    base_module = ModuleType("sica.base")
    base_module.whitening = whitening
    base_module.StabilizedICA = FakeStabilizedICA
    sica_module.base = base_module
    monkeypatch.setitem(sys.modules, "sica", sica_module)
    monkeypatch.setitem(sys.modules, "sica.base", base_module)

    pp.mstd(
        adata,
        m=2,
        M=2,
        n_runs=1,
        use_highly_variable=False,
        space="cell",
        show_progress=False,
    )

    np.testing.assert_array_equal(whitening_inputs[0], adata.X)
    np.testing.assert_array_equal(fitted_matrices[0], adata.X[:, :2].T)
    assert adata.uns["ica"]["mstd"]["params"]["space"] == "cell"


def test_stabilized_ica_and_mstd_store_h5ad_compatible_results(tmp_path, capsys):
    pytest.importorskip("sica")
    adata = make_adata()

    result = pp.mstd(
        adata,
        m=2,
        M=4,
        step=2,
        n_runs=2,
        algorithm="fastica_par",
        max_iter=100,
        show_progress=False,
    )
    assert result is None
    captured = capsys.readouterr()
    assert captured.out == ""
    assert adata.uns["ica"]["mstd"]["component_counts"].tolist() == [2, 4]
    fig, axes = pl.plot_all_stability_indices(adata, return_fig=True)
    assert axes.get_ylim() == (0.0, 1.0)
    assert len(axes.lines) == 3
    np.testing.assert_array_equal(axes.lines[0].get_xdata(), [1, 2])
    np.testing.assert_array_equal(axes.lines[1].get_xdata(), [1, 2, 3, 4])
    assert all(line.get_marker() == "None" for line in axes.lines[:2])
    plotted_elbow = adata.uns["ica"]["mstd"]["optimal_n_compt"]
    np.testing.assert_array_equal(
        axes.lines[-1].get_xdata(), [plotted_elbow, plotted_elbow]
    )
    import matplotlib.pyplot as plt

    plt.close(fig)

    pp.ica(
        adata,
        n_runs=2,
        algorithm="fastica_par",
        max_iter=100,
        n_top_genes=4,
    )
    captured = capsys.readouterr()
    selected = adata.uns["ica"]["mstd"]["optimal_n_compt"]
    assert selected in {2, 4}
    assert pp.select_mstd_components(adata, last_n=2) == selected
    assert f"'optimal_n_compt': {selected}" in captured.out
    assert adata.obsm["X_ica"].shape == (adata.n_obs, selected)
    assert adata.varm["ICs"].shape == (adata.n_vars, selected)
    assert len(adata.uns["ica"]["stability_indexes"]) == selected
    assert pp.select_stable_components(adata, min_stability=0).tolist() == list(
        range(1, selected + 1)
    )

    fig, axis = pl.plot_mean_stability_index(adata, return_fig=True)
    assert axis.get_ylim() == (0.0, 1.0)
    assert any(
        np.array_equal(line.get_xdata(), [selected, selected])
        for line in axis.lines
    )
    plt.close(fig)

    pp.ica(
        adata,
        n_comps=3,
        n_runs=2,
        algorithm="fastica_par",
        max_iter=100,
        n_top_genes=4,
        verbose=False,
    )
    assert capsys.readouterr().out == ""
    assert adata.obsm["X_ica"].shape == (adata.n_obs, 3)

    output = tmp_path / "stabilized_ica.h5ad"
    adata.write_h5ad(output, compression="gzip")
    reopened = anndata.read_h5ad(output)
    assert reopened.uns["ica"]["mstd"]["stability_indexes"].shape == (2, 4)
    assert reopened.obsm["X_ica"].shape == (adata.n_obs, 3)


def test_stability_plots_show_by_default_and_can_return_figures(monkeypatch):
    import matplotlib.pyplot as plt

    adata = make_adata(n_obs=10, n_vars=5)
    adata.uns["ica"] = {
        "mstd": {
            "component_counts": np.array([2, 4]),
            "stability_indexes": np.array(
                [[0.8, 0.7, np.nan, np.nan], [0.9, 0.8, 0.7, 0.6]]
            ),
            "optimal_n_compt": 2,
        }
    }
    shown = []
    monkeypatch.setattr(plt, "show", lambda: shown.append(True))

    assert pl.plot_all_stability_indices(adata) is None
    assert pl.plot_mean_stability_index(adata) is None
    assert shown == [True, True]

    fig, axis = pl.plot_all_stability_indices(
        adata, optimal_n_compt=3, return_fig=True
    )
    np.testing.assert_array_equal(axis.lines[-1].get_xdata(), [3, 3])
    assert axis.lines[-1].get_linestyle() == "--"
    plt.close(fig)

    fig, axis = pl.plot_all_stability_indices(
        adata, optimal_n_compt=None, return_fig=True
    )
    assert len(axis.lines) == 2
    plt.close(fig)


def test_drop_components_from_ica_embedding_without_pca_metadata():
    adata = make_adata(n_obs=10, n_vars=5)
    adata.obsm["X_ica"] = np.arange(40).reshape(10, 4)
    adata.varm["ICs"] = np.arange(20).reshape(5, 4)

    keys = pp.drop_components_from_embedding(
        adata,
        [2],
        new_suffix="no_cell_cycle",
    )

    assert keys == {
        "obsm": "X_ica_no_cell_cycle",
        "varm": "ICs_no_cell_cycle",
        "variance_ratio": None,
    }
    np.testing.assert_array_equal(
        adata.obsm[keys["obsm"]], adata.obsm["X_ica"][:, [0, 2, 3]]
    )
    np.testing.assert_array_equal(
        adata.varm[keys["varm"]], adata.varm["ICs"][:, [0, 2, 3]]
    )


def test_drop_components_uses_thresholds_saved_by_association_plot():
    adata = make_adata(n_obs=100, n_vars=8, seed=31)
    rng = np.random.default_rng(32)
    embedding = rng.normal(size=(adata.n_obs, 3))
    embedding[:, 0] = np.linspace(-1, 1, adata.n_obs)
    adata.obsm["X_ica"] = embedding
    adata.varm["ICs"] = rng.normal(size=(adata.n_vars, 3))
    adata.obs["cell_cycle_score"] = (
        2 * embedding[:, 0] + rng.normal(scale=0.01, size=adata.n_obs)
    )
    pp.compute_score_component_associations(
        adata, "cell_cycle_score", compute_mutual_info=False
    )

    fig, _ = pl.plot_score_component_associations(
        adata,
        "cell_cycle_score",
        correlation_threshold=0.8,
        mutual_info_threshold=None,
    )
    import matplotlib.pyplot as plt

    plt.close(fig)
    thresholds = adata.uns["ica"]["associations"]["cell_cycle_score"]["thresholds"]
    assert thresholds == {
        "correlation_threshold": 0.8,
        "mutual_info_threshold": None,
    }

    keys = pp.drop_components_from_embedding(
        adata, score_keys=["cell_cycle_score"], new_suffix="cell_cycle_removed"
    )
    assert adata.obsm[keys["obsm"]].shape == (adata.n_obs, 2)
    assert keys["obsm"] == "X_ica_cell_cycle_removed"
    assert keys["varm"] == "ICs_cell_cycle_removed"
    np.testing.assert_array_equal(
        adata.obsm[keys["obsm"]], adata.obsm["X_ica"][:, 1:]
    )


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
    assert all(axis.get_ylim() == (0.0, 1.0) for axis in axes)
    assert adata.uns["ica"]["associations"]["score"]["thresholds"] == {
        "correlation_threshold": 0.3,
        "mutual_info_threshold": None,
    }
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
    assert reopened.uns["ica"]["associations"]["score"]["thresholds"][
        "mutual_info_threshold"
    ] is None


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
