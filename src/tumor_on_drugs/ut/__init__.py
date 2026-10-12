"""General-purpose utilities for analysis workflows."""

from .environment import check_prepare_gpu, in_notebook
from .io import read_h5ad_dropping_incompatible, stream_matrix_to_pipe

__all__ = [
    "check_prepare_gpu",
    "in_notebook",
    "read_h5ad_dropping_incompatible",
    "stream_matrix_to_pipe",
]
