"""Core statistical modules of the certifier.

The four modules import one another by their original top-level names
(``assignment_oracle``, ``best_alternative``, ``finite_ray_model``,
``track_and_certify_general``).  To keep them byte-identical, this package
loads them from files in dependency order and registers them in
``sys.modules`` under those names -- **without** touching ``sys.path``, so
nothing else on the import path is shadowed.

If one of these names is already imported from somewhere else (a user module
with the same name), loading raises ``ImportError`` immediately with an
explanation instead of silently mixing implementations.  Each module is also
registered under ``track_certify._vendor.<name>`` for unambiguous access.
"""

import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ORDER = (
    "assignment_oracle",
    "best_alternative",
    "finite_ray_model",
    "track_and_certify_general",
)


def _load(name):
    path = os.path.join(_HERE, name + ".py")
    existing = sys.modules.get(name)
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if existing_file is None or \
                os.path.dirname(os.path.abspath(existing_file)) != _HERE:
            raise ImportError(
                f"track-certify provides a core module named {name!r}, "
                f"but a different module with that name is already imported "
                f"from {existing_file!r}. Import track_certify before the "
                f"conflicting module, or rename it.")
        return existing
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        del sys.modules[name]
        raise
    return module


for _name in _ORDER:
    _mod = _load(_name)
    sys.modules[__name__ + "." + _name] = _mod
    globals()[_name] = _mod

del _name, _mod
