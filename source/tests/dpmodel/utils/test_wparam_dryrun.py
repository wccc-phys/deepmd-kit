# SPDX-License-Identifier: LGPL-3.0-or-later
"""Dry-run: adding a NEW fitting parameter (``wparam``) to the registry.

This is the acceptance proof for the FittingParams design: appending ONE
``ParamSpec`` is enough for the container, the absorption chain, the data
requirement builder and the metadata writer to recognise the new parameter
-- no per-file edits anywhere else.
"""

import numpy as np
import pytest

from deepmd.dpmodel.utils.fitting_params import (
    FittingParams,
    ParamSpec,
)

import importlib.util
import pathlib as _pl

_spec = _pl.Path(__file__).with_name('test_fitting_params.py')
import importlib.util as _iu
_spec_loader = _iu.spec_from_file_location('test_fitting_params', _spec)
_mod = _iu.module_from_spec(_spec_loader)
_spec_loader.loader.exec_module(_mod)
FakeModel = _mod.FakeModel
_item_dict = _mod._item_dict


class WparamModel(FakeModel):
    """A model that grew a ``wparam`` conditioning input."""

    def __init__(self, wparam_dim=2, wparam_default=None, **kw):
        super().__init__(**kw)
        self.dims["wparam"] = wparam_dim
        if wparam_default is not None:
            self.defaults["wparam"] = wparam_default

    def get_dim_wparam(self) -> int:
        return self.dims.get("wparam", 0)

    def has_default_wparam(self) -> bool:
        return "wparam" in self.defaults

    def get_default_wparam(self):
        return self.defaults.get("wparam")


@pytest.fixture()
def wparam_registry():
    """Temporarily append the wparam spec to the registry."""
    spec = ParamSpec(
        key="wparam",
        frame_level=True,
        dim_getter="get_dim_wparam",
        has_default_getter="has_default_wparam",
        default_getter="get_default_wparam",
        metadata_keys=("dim_wparam", "has_default_wparam", "default_wparam"),
        optional=True,
    )
    old = FittingParams.SPEC
    FittingParams.SPEC = old + (spec,)
    yield spec
    FittingParams.SPEC = old


def test_new_parameter_container(wparam_registry):
    fp = FittingParams(wparam=np.zeros((1, 2)))
    assert fp.wparam is not None
    with pytest.raises(ValueError):
        FittingParams(not_a_param=1)


def test_new_parameter_absorbs(wparam_registry):
    cond = FittingParams(wparam=np.ones((1, 2)))
    fparam, uparam, aparam, charge_spin, wparam = cond.absorb(
        fparam=None, uparam=None, aparam=None, charge_spin=None
    )
    assert wparam is not None and wparam.shape == (1, 2)
    assert fparam is None  # nothing set for it


def test_new_parameter_requirement(wparam_registry):
    from deepmd.utils.data import (
        DataRequirementItem,
    )

    model = WparamModel(wparam_dim=2, wparam_default=[0.5, 0.25])
    items = FittingParams.requirement_items(model, DataRequirementItem)
    keys = [i.key for i in items]
    # registry order with wparam appended at the end
    assert keys[-1] == "wparam"
    witem = _item_dict(items[-1])
    assert witem["ndof"] == 2
    assert witem["atomic"] is False  # frame_level
    assert witem["must"] is False  # has a default
    assert witem["source_policy"] == "default"


def test_new_parameter_metadata(wparam_registry):
    model = WparamModel(wparam_dim=2)
    meta = FittingParams.metadata_dict(model)
    assert meta["dim_wparam"] == 2
    assert meta["has_default_wparam"] is False
    assert meta["default_wparam"] is None
