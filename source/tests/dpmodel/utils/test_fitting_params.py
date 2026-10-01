# SPDX-License-Identifier: LGPL-3.0-or-later
"""FittingParams registry: golden-parity tests against the legacy per-parameter
implementations, plus the container/split/prune/prepare semantics.

The golden reference for data requirements is the (pre-refactor)
``get_additional_data_requirement`` chain; for metadata, the per-parameter
keys written by ``_collect_metadata``; for lower-input preparation, the
``_prepare_optional_lower_inputs`` branch semantics.
"""

import numpy as np
import pytest

from deepmd.dpmodel.utils.fitting_params import (
    FittingParams,
)


class FakeModel:
    """Probes every getter the registry may call."""

    def __init__(
        self,
        *,
        dims: dict[str, int] | None = None,
        defaults: dict[str, object] | None = None,
        uparam_mode: str = "frame",
        var_name: str | None = None,
        has_spin: bool = False,
        chg_spin_ebd: bool = False,
    ) -> None:
        self.dims = dims or {}
        self.defaults = defaults or {}
        self._uparam_mode = uparam_mode
        self._var_name = var_name
        self._has_spin = has_spin
        self._chg_spin_ebd = chg_spin_ebd

    # generic getters -------------------------------------------------
    def get_dim_fparam(self) -> int:
        return self.dims.get("fparam", 0)

    def get_dim_uparam(self) -> int:
        return self.dims.get("uparam", 0)

    def get_dim_aparam(self) -> int:
        return self.dims.get("aparam", 0)

    def get_dim_chg_spin(self) -> int:
        return self.dims.get("charge_spin", 0)

    def has_default_fparam(self) -> bool:
        return "fparam" in self.defaults

    def get_default_fparam(self):
        return self.defaults.get("fparam")

    def has_default_uparam(self) -> bool:
        return "uparam" in self.defaults

    def get_default_uparam(self):
        return self.defaults.get("uparam")

    def get_uparam_mode(self) -> str:
        return self._uparam_mode

    def has_chg_spin_ebd(self) -> bool:
        return self._chg_spin_ebd

    def get_default_chg_spin(self):
        return self.defaults.get("charge_spin")

    def get_var_name(self):
        return self._var_name

    def has_spin(self) -> bool:
        return self._has_spin


# ---------------------------------------------------------------------------
# registry order = ABI contract
# ---------------------------------------------------------------------------


def test_registry_order_is_the_export_abi():
    assert FittingParams.keys() == ("fparam", "uparam", "aparam", "charge_spin")


def test_from_legacy_absorbs_known_ignores_unknown():
    fp = FittingParams.from_legacy(
        fparam=np.zeros((1, 1)), uparam=None, charge_spin=None, do_atomic_virial=True
    )
    assert fp.fparam is not None
    assert fp.uparam is None
    assert fp.charge_spin is None


def test_container_rejects_unknown_parameters():
    with pytest.raises(ValueError, match="wparam"):
        FittingParams(wparam=1)


# ---------------------------------------------------------------------------
# data requirements: golden parity with the legacy chain
# ---------------------------------------------------------------------------


def _item_dict(item):
    return {
        "key": item.key,
        "ndof": item.ndof,
        "atomic": item.atomic,
        "must": item.must,
        "high_prec": item.high_prec,
        "default": item.default if isinstance(item.default, float) else np.asarray(item.default).tolist(),
        "source_policy": item.source_policy,
    }


def _legacy_requirement_items(model):
    """The pre-refactor pt_expt chain, verbatim, restricted to registered
    parameters (the legacy spin block stays outside the registry)."""
    from deepmd.utils.data import (
        DataRequirementItem,
    )

    reqs = []
    if model.get_dim_fparam() > 0:
        has_default = model.has_default_fparam()
        dflt = np.asarray(model.get_default_fparam()) if has_default else 0.0
        reqs.append(
            DataRequirementItem(
                "fparam",
                model.get_dim_fparam(),
                atomic=False,
                must=not has_default,
                default=dflt,
                source_policy="default" if has_default else "tracked",
            )
        )
    if model.get_dim_uparam() > 0:
        has_default = model.has_default_uparam()
        dflt = np.asarray(model.get_default_uparam()) if has_default else 0.0
        reqs.append(
            DataRequirementItem(
                "uparam",
                model.get_dim_uparam(),
                atomic=model.get_uparam_mode() == "atomic",
                must=not has_default,
                default=dflt,
                source_policy="default" if has_default else "tracked",
            )
        )
    if model.get_dim_aparam() > 0:
        reqs.append(
            DataRequirementItem(
                "aparam", model.get_dim_aparam(), atomic=True, must=True
            )
        )
    if model.has_chg_spin_ebd():
        dflt = model.get_default_chg_spin()
        has_default = dflt is not None
        if has_default:
            dflt = np.asarray(dflt)
        else:
            dflt = 0.0
        reqs.append(
            DataRequirementItem(
                "charge_spin",
                ndof=2,
                atomic=False,
                must=not has_default,
                default=dflt,
                source_policy="default" if has_default else "tracked",
            )
        )
    return reqs


@pytest.mark.parametrize(
    "model",
    [
        FakeModel(),  # nothing active
        FakeModel(dims={"fparam": 1}),  # fparam, no default
        FakeModel(dims={"fparam": 2}, defaults={"fparam": [300.0, 1.0]}),
        FakeModel(dims={"uparam": 1}, defaults={"uparam": 3.4}),
        FakeModel(dims={"uparam": 1}, uparam_mode="atomic"),
        FakeModel(dims={"aparam": 5}),
        FakeModel(dims={"charge_spin": 2}, chg_spin_ebd=True),
        FakeModel(dims={"charge_spin": 2}, chg_spin_ebd=True, defaults={"charge_spin": [1.0, 2.0]}),
        FakeModel(  # everything at once
            dims={"fparam": 1, "uparam": 1, "aparam": 3, "charge_spin": 2},
            defaults={"fparam": [300.0], "uparam": 3.4, "charge_spin": [1.0, 2.0]},
            chg_spin_ebd=True,
        ),
    ],
)
def test_requirement_items_match_legacy(model):
    from deepmd.utils.data import (
        DataRequirementItem,
    )

    new = FittingParams.requirement_items(model, DataRequirementItem)
    old = _legacy_requirement_items(model)
    assert [_item_dict(i) for i in new] == [_item_dict(i) for i in old]


# ---------------------------------------------------------------------------
# metadata: golden parity
# ---------------------------------------------------------------------------


def test_metadata_dict_covers_all_legacy_keys():
    model = FakeModel(
        dims={"fparam": 1, "uparam": 1, "aparam": 2, "charge_spin": 2},
        defaults={"fparam": [300.0], "uparam": 3.4, "charge_spin": [1.0, 2.0]},
        uparam_mode="atomic",
        chg_spin_ebd=True,
    )
    meta = FittingParams.metadata_dict(model)
    for k in (
        "dim_fparam",
        "has_default_fparam",
        "default_fparam",
        "dim_uparam",
        "uparam_mode",
        "has_default_uparam",
        "default_uparam",
        "dim_aparam",
        "dim_chg_spin",
        "has_chg_spin_ebd",
        "has_default_chg_spin",
        "default_chg_spin",
        "chg_spin_table_ranges",
    ):
        assert k in meta, k
    assert meta["dim_fparam"] == 1
    assert meta["uparam_mode"] == "atomic"
    assert meta["has_default_uparam"] is True
    assert meta["default_uparam"] == 3.4
    assert meta["chg_spin_table_ranges"] is None


# ---------------------------------------------------------------------------
# split / prune
# ---------------------------------------------------------------------------


def test_split_moves_uparam_to_label_for_mlu():
    batch = {
        "coord": 1,
        "uparam": 2,
        "find_uparam": True,
        "energy": 3,
        "find_energy": True,
    }
    mlu = FakeModel(var_name="uparam")  # dim_uparam == 0
    inputs, labels = FittingParams.split(batch, mlu)
    assert "uparam" in labels and "uparam" not in inputs

    plain = FakeModel(var_name="energy", dims={"uparam": 1})
    inputs, labels = FittingParams.split(batch, plain)
    assert "uparam" in inputs and "uparam" not in labels


def test_prune_optionals_matches_legacy_loop():
    inputs = {"fparam": 1, "uparam": 2, "aparam": 3, "charge_spin": 4}
    labels = {"find_fparam": False, "find_uparam": True, "find_charge_spin": False}
    FittingParams.prune_optionals(inputs, labels)
    assert "fparam" not in inputs
    assert "uparam" in inputs
    assert "charge_spin" not in inputs
    assert "aparam" in inputs  # never pruned


# ---------------------------------------------------------------------------
# prepare_lower_inputs semantics
# ---------------------------------------------------------------------------


def _to_tensor(key, value, atomic):
    arr = np.asarray(value, dtype=np.float64)
    if key == "uparam" and not atomic:
        return np.repeat(arr, 4, axis=0)  # frame->node, 4 atoms/frame
    if atomic:
        return arr.reshape(-1, arr.shape[-1])
    return arr


def test_prepare_frame_param_given():
    meta = {"dim_fparam": 1}
    out = FittingParams.prepare_lower_inputs(
        {"fparam": [[300.0]]}, FakeModel(dims={"fparam": 1}), meta, 1, 4, _to_tensor
    )
    assert out["fparam"].shape == (1, 1)


def test_prepare_frame_param_default_fill():
    meta = {"dim_fparam": 1, "has_default_fparam": True, "default_fparam": [300.0]}
    out = FittingParams.prepare_lower_inputs(
        {"fparam": None}, FakeModel(dims={"fparam": 1}), meta, 2, 4, _to_tensor
    )
    # broadcast over frames, mirroring the legacy expand(nframes, -1)
    assert out["fparam"].shape == (2, 1)
    assert float(out["fparam"].reshape(-1)[0]) == 300.0


def test_prepare_frame_param_missing_raises():
    meta = {"dim_fparam": 1}
    with pytest.raises(ValueError, match="fparam is required"):
        FittingParams.prepare_lower_inputs(
            {"fparam": None}, FakeModel(dims={"fparam": 1}), meta, 1, 4, _to_tensor
        )


def test_prepare_aparam_required_no_default():
    meta = {"dim_aparam": 2}
    with pytest.raises(ValueError, match="aparam is required"):
        FittingParams.prepare_lower_inputs(
            {"aparam": None}, FakeModel(dims={"aparam": 2}), meta, 1, 4, _to_tensor
        )


def test_prepare_atomic_uparam_flattens_to_nodes():
    values = {"uparam": np.arange(16).reshape(1, 16, 1)}
    model = FakeModel(dims={"uparam": 1}, uparam_mode="atomic")
    out = FittingParams.prepare_lower_inputs(
        values, model, {"dim_uparam": 1}, 1, 16, _to_tensor
    )
    assert out["uparam"].shape == (16, 1)


def test_prepare_inactive_is_none():
    out = FittingParams.prepare_lower_inputs(
        {"uparam": None}, FakeModel(), {"dim_uparam": 0}, 1, 4, _to_tensor
    )
    assert out["uparam"] is None


def test_prepare_charge_spin_default():
    meta = {
        "dim_chg_spin": 2,
        "has_default_chg_spin": True,
        "default_chg_spin": [1.0, 2.0],
    }
    out = FittingParams.prepare_lower_inputs(
        {"charge_spin": None}, FakeModel(dims={"charge_spin": 2}), meta, 1, 4, _to_tensor
    )
    assert out["charge_spin"].shape == (1, 2)
