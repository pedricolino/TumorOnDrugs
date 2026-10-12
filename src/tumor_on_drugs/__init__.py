"""Single-cell analysis and visualization helpers."""

from importlib import import_module
from typing import TYPE_CHECKING

__version__ = "0.1.0"

if TYPE_CHECKING:
    from . import pl, pp, tl, ut

_SUBMODULES = {
    "pl": "tumor_on_drugs.pl",
    "pp": "tumor_on_drugs.pp",
    "tl": "tumor_on_drugs.tl",
    "ut": "tumor_on_drugs.ut",
}

__all__ = ["__version__", *_SUBMODULES]


def __dir__():
    return sorted(__all__)


def __getattr__(name):
    """Load a public namespace only when it is first accessed."""
    if name in _SUBMODULES:
        module = import_module(_SUBMODULES[name])
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")