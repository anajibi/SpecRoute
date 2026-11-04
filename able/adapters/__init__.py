from .base import Adapter, AttrSpec
from . import c3di, celebahq

_BY_NAME = {"c3di": c3di.C3DIAdapter, "celebahq": celebahq.CelebAHQAdapter}


def for_dataset(name: str) -> type:
    if name not in _BY_NAME:
        raise KeyError(f"no adapter for dataset {name!r}; have {', '.join(_BY_NAME)}")
    return _BY_NAME[name]
