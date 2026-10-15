from pathlib import Path
from types import SimpleNamespace
import sys

import tumor_on_drugs
from tumor_on_drugs import pl, pp, tl, ut


def test_public_package_import_is_lightweight():
    assert tumor_on_drugs.__version__ == "0.1.0"
    assert set(tumor_on_drugs.__all__) == {"__version__", "pl", "pp", "tl", "ut"}


def test_plot_helper_is_publicly_exported():
    assert "split_dim_plot" in pl.__all__


def test_scanpy_style_modules_expose_classified_helpers():
    assert "safe_score_genes" in pp.__all__
    assert "umap_based_on_paga" in tl.__all__
    assert "split_dim_plot" in pl.__all__


def test_notebook_helpers_are_package_exports():
    assert callable(pp.drop_components_from_embedding)
    assert callable(pp.robustica)
    assert callable(pp.stabilized_ica)
    assert not hasattr(pp, "ica")
    assert callable(pl.color_cells)
    assert callable(tl.compute_condition_network)
    assert callable(tl.build_ego_graph)


def test_utility_module_exposes_runtime_helpers():
    assert callable(ut.in_notebook)
    assert callable(ut.check_prepare_gpu)
    assert callable(ut.read_h5ad_dropping_incompatible)
    assert callable(ut.stream_matrix_to_pipe)


def test_read_h5ad_dropping_incompatible_uses_and_cleans_a_copy(tmp_path, monkeypatch):
    source = tmp_path / "sample.h5ad"
    source.write_bytes(b"original")
    backup = Path(f"{source}.bak")
    backup.write_bytes(b"existing backup")
    removed_keys = {"/uns/legacy"}
    read_paths = []
    adata = object()

    class FakeH5File:
        def __init__(self, path, mode):
            self.path = Path(path)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def __contains__(self, key):
            return key in removed_keys

        def __delitem__(self, key):
            removed_keys.remove(key)

    def read_h5ad(path):
        read_paths.append(Path(path))
        if Path(path) == source or removed_keys:
            raise ValueError("Could not read '/uns/legacy'")
        return adata

    monkeypatch.setitem(sys.modules, "h5py", SimpleNamespace(File=FakeH5File))
    monkeypatch.setitem(sys.modules, "scanpy", SimpleNamespace(read_h5ad=read_h5ad))

    result, removed = ut.read_h5ad_dropping_incompatible(source, verbose=False)

    assert result is adata
    assert removed == ["/uns/legacy"]
    assert source.read_bytes() == b"original"
    assert backup.read_bytes() == b"existing backup"
    assert read_paths[1] != source
    assert not read_paths[1].exists()