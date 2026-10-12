"""General-purpose utility implementations."""

import os
import re
import shutil
import tempfile
from pathlib import Path


def stream_matrix_to_pipe(matrix_df, pipe_path):
    """Write a DataFrame to a named pipe for a streaming consumer."""
    with open(pipe_path, "w") as pipe:
        matrix_df.to_csv(pipe, sep="\t")


def read_h5ad_dropping_incompatible(h5ad_path, max_iter=50, verbose=True):
    """Read an h5ad file, removing incompatible elements from a temporary copy.

    The original file is never modified. The returned AnnData object is loaded
    into memory before the temporary copy is deleted.

    Returns
    -------
    adata : AnnData
        The loaded object.
    removed : list[str]
        HDF5 keys removed from the temporary copy.
    """
    import h5py
    import scanpy as sc

    h5ad_path = Path(h5ad_path)

    try:
        return sc.read_h5ad(h5ad_path), []
    except Exception as read_err:
        err = read_err
        if verbose:
            print(f"Failed to read {h5ad_path}: {err}")

    fd, temp_name = tempfile.mkstemp(
        prefix=f".{h5ad_path.name}.",
        suffix=".h5ad",
        dir=h5ad_path.parent,
    )
    os.close(fd)
    copy_path = Path(temp_name)
    removed = []

    try:
        shutil.copyfile(h5ad_path, copy_path)
        for i in range(max_iter):
            match = re.search(r"'(.*?)'", str(err))
            if not match:
                raise RuntimeError(
                    f"Could not identify incompatible key from error: {err}"
                ) from err
            key = match.group(1)

            if key in removed:
                raise RuntimeError(
                    f"Key '{key}' was already removed but error persists: {err}"
                ) from err

            with h5py.File(copy_path, "a") as h5_file:
                candidates = [key, f"layers/{key}"]
                target = next((candidate for candidate in candidates if candidate in h5_file), None)
                if target is None:
                    raise RuntimeError(
                        f"Incompatible key '{key}' not found in copied file."
                    ) from err
                del h5_file[target]
                removed.append(target)
                if verbose:
                    print(f"[iter {i + 1}] Removed '{target}' from copy")

            try:
                adata = sc.read_h5ad(copy_path)
                if verbose:
                    print(f"Read succeeded after removing: {removed}")
                return adata, removed
            except Exception as read_err:
                err = read_err
                if verbose:
                    print(f"Still failing: {err}")

        raise RuntimeError(f"Gave up after {max_iter} iterations. Removed: {removed}")
    finally:
        if copy_path.exists():
            copy_path.unlink()
            if verbose:
                print(f"Deleted copy {copy_path}")
