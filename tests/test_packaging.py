"""Packaging hygiene: no sys.path pollution, explicit collision errors."""
import os
import sys
import types

import pytest


def test_no_sys_path_pollution():
    import track_certify  # noqa: F401
    vendor_dir = os.path.dirname(
        sys.modules["track_certify._vendor"].__file__)
    assert vendor_dir not in sys.path


def test_vendored_names_registered_and_aliased():
    import track_certify  # noqa: F401
    for name in ("assignment_oracle", "best_alternative",
                 "finite_ray_model", "track_and_certify_general"):
        assert name in sys.modules
        assert sys.modules["track_certify._vendor." + name] \
            is sys.modules[name]
        f = sys.modules[name].__file__
        assert os.path.basename(os.path.dirname(f)) == "_vendor"


def test_collision_raises_clear_error():
    from track_certify import _vendor
    fake = types.ModuleType("finite_ray_model")
    fake.__file__ = "/somewhere/else/finite_ray_model.py"
    saved = sys.modules["finite_ray_model"]
    sys.modules["finite_ray_model"] = fake
    try:
        with pytest.raises(ImportError, match="already imported"):
            _vendor._load("finite_ray_model")
    finally:
        sys.modules["finite_ray_model"] = saved
