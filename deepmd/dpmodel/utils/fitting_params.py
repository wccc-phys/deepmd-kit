# SPDX-License-Identifier: LGPL-3.0-or-later
"""Fitting-conditioning parameter registry and runtime container.

Every model in deepmd-kit consumes a fixed set of *fitting parameters* --
per-frame or per-atom conditioning tensors fed next to coordinates:

* ``fparam``      frame-level parameters (e.g. temperature);
* ``uparam``      DFT+U Hubbard-U parameters (frame or atomic mode);
* ``aparam``      per-atom parameters (always required when active);
* ``charge_spin`` frame-level charge/spin FiLM conditioning.

Each parameter used to be threaded through the code base as an individual
named argument, so adding a new one meant touching ~40 signatures and ~8
per-parameter branching sites (data requirements, batch splitting, optional
pruning, inference default-filling, export ABI slot assembly, C++ input
assembly).  This module collapses that knowledge into ONE ordered registry:

``ParamSpec``
    Static description of one parameter: which model getters report its
    dimension/default, whether it is frame- or atom-level, whether it can
    be pruned when absent, and how it appears in exported metadata.

``FittingParams``
    (1) a runtime *value container* -- ``model(coord, atype, box,
    cond=fp_obj)`` with legacy keyword absorption (``from_legacy``); and
    (2) the static *spec source* -- every per-parameter branch in the
    data / trainer / inference / export layers is driven from
    ``FittingParams.SPEC`` instead of hand-written per-parameter code.

Adding a future parameter (say ``wparam``) = append one ``ParamSpec`` to
``FittingParams.SPEC`` + add the corresponding ``get_dim_wparam`` /
``get_default_wparam`` getters on the model/fitting.  Every layer that
consumes this registry picks it up automatically.

The ordered registry IS the ABI contract: ``SPEC`` order defines the tail
slot order of the graph-lower export ``(fparam, uparam, aparam,
charge_spin)`` and the C++ input assembly order.  Appending a new spec at
the END is backward compatible; inserting in the middle changes the
exported ABI and is a breaking action.

``spin`` is deliberately NOT a spec: it is a descriptor capability with
its own autograd-leaf role in the graph lower (force_mag), not a plain
conditioning input.
"""

from dataclasses import (
    dataclass,
    field,
)
from typing import (
    Any,
    Callable,
)

import numpy as np

__all__ = [
    "FittingParams",
    "ParamSpec",
]


@dataclass(frozen=True)
class ParamSpec:
    """Static description of one fitting parameter.

    Attributes
    ----------
    key
        Parameter name: the ``DataRequirementItem`` key, the batch/input
        dict key, the export kwarg name and the ``FittingParams`` attribute
        name -- all the same string.
    frame_level
        ``True`` when the runtime tensor is per-frame ``(nf, nd)``;
        ``False`` when it is per-atom ``(nf, natoms, nd)``.
    dim_getter
        Name of the model/fitting getter returning the parameter dimension
        (e.g. ``"get_dim_fparam"``).  The generic metadata/metadata-only
        paths probe ``hasattr`` before calling.
    has_default_getter, default_getter
        Names of the getters reporting the fallback value, or ``None`` for
        parameters without a default (``aparam`` is always required).
    metadata_keys
        Exported-metadata keys written for this parameter, in order.
    optional
        Whether the parameter may be dropped from the model inputs when its
        ``find_<key>`` flag is ``False`` (frame parameters and charge_spin
        can; ``aparam`` may not).
    atomic_mode_getter
        Name of a getter returning the axis mode for parameters whose
        runtime layout switches between frame- and node-level (only
        ``uparam`` has ``get_uparam_mode`` -> "atomic"); ``None`` otherwise.
    """

    key: str
    frame_level: bool
    dim_getter: str
    has_default_getter: str | None = None
    default_getter: str | None = None
    metadata_keys: tuple[str, ...] = ()
    optional: bool = True
    atomic_mode_getter: str | None = None
    # C++ scalar member on DeepPotPTExpt holding the dimension, used by the
    # CondParams.h assembly loop (kept here so python and C++ share one
    # ordering contract).
    cpp_dim_member: str = ""
    # Activation getter for parameters whose presence is decided by a
    # capability rather than by their dimension (charge_spin ->
    # ``has_chg_spin_ebd``); ``None`` -> active iff dim > 0.
    active_getter: str | None = None

    def call_getter(self, obj: Any, name: str, default: Any = None) -> Any:
        """Call ``getattr(obj, name)()`` when the attribute exists."""
        getter = getattr(obj, name, None)
        if getter is None or not callable(getter):
            return default
        return getter()


_FPARAM = ParamSpec(
    key="fparam",
    frame_level=True,
    dim_getter="get_dim_fparam",
    has_default_getter="has_default_fparam",
    default_getter="get_default_fparam",
    metadata_keys=("dim_fparam", "has_default_fparam", "default_fparam"),
    optional=True,
    cpp_dim_member="dfparam",
)

_UPARAM = ParamSpec(
    key="uparam",
    frame_level=True,  # frame mode default; atomic mode via atomic_mode_getter
    dim_getter="get_dim_uparam",
    has_default_getter="has_default_uparam",
    default_getter="get_default_uparam",
    metadata_keys=(
        "dim_uparam",
        "uparam_mode",
        "has_default_uparam",
        "default_uparam",
    ),
    optional=True,
    atomic_mode_getter="get_uparam_mode",
    cpp_dim_member="duparam",
)

_APARAM = ParamSpec(
    key="aparam",
    frame_level=False,
    dim_getter="get_dim_aparam",
    has_default_getter=None,
    default_getter=None,
    metadata_keys=("dim_aparam",),
    optional=False,
    cpp_dim_member="daparam",
)

_CHARGE_SPIN = ParamSpec(
    key="charge_spin",
    frame_level=True,
    dim_getter="get_dim_chg_spin",
    has_default_getter="has_default_chg_spin",
    default_getter="get_default_chg_spin",
    metadata_keys=(
        "dim_chg_spin",
        "has_chg_spin_ebd",
        "has_default_chg_spin",
        "default_chg_spin",
    ),
    optional=True,
    cpp_dim_member="dchgspin",
    active_getter="has_chg_spin_ebd",
)


def _default_metadata_getter(spec: ParamSpec, model: Any) -> dict[str, Any]:
    """Generic metadata writer: dim / mode / has_default / default keys."""
    out: dict[str, Any] = {}
    dim = spec.call_getter(model, spec.dim_getter, 0)
    out[spec.metadata_keys[0]] = int(dim or 0)
    rest = list(spec.metadata_keys[1:])
    if spec.atomic_mode_getter is not None:
        mode_getter = getattr(model, spec.atomic_mode_getter, None)
        out[rest.pop(0)] = (
            str(mode_getter()) if mode_getter is not None and callable(mode_getter) else "frame"
        )
    if spec.has_default_getter is not None:
        has_default = bool(spec.call_getter(model, spec.has_default_getter, False))
        out[rest.pop(0)] = has_default
        default = spec.call_getter(model, spec.default_getter) if spec.default_getter else None
        out[rest.pop(0)] = default
    assert not rest, f"unmapped metadata keys for {spec.key}: {rest}"
    return out


def _charge_spin_metadata(model: Any) -> dict[str, Any]:
    """charge_spin extras that do not fit the generic getter pattern.

    ``has_chg_spin_ebd`` is a model capability (embedding-table conditioning)
    rather than a plain default-presence flag, and the table ranges are
    written next to the defaults.
    """
    has_ebd = bool(model.has_chg_spin_ebd())
    default = model.get_default_chg_spin()
    ranges = None
    get_ranges = getattr(model, "get_chg_spin_table_ranges", None)
    if callable(get_ranges):
        ranges = get_ranges()
    if ranges is None:
        get_ranges2 = getattr(model, "chg_spin_table_ranges", None)
        ranges = get_ranges2() if callable(get_ranges2) else None
    return {
        "dim_chg_spin": int(model.get_dim_chg_spin() or 0),
        "has_chg_spin_ebd": has_ebd,
        "has_default_chg_spin": default is not None,
        "default_chg_spin": default,
        "chg_spin_table_ranges": (
            [list(b) for b in ranges] if ranges is not None else None
        ),
    }


class FittingParams:
    """Ordered registry + runtime container of the fitting parameters.

    As a container, an instance holds one optional tensor per spec
    (``fp.fparam``, ``fp.uparam``, ...).  As a spec source, the class
    exposes the registry (``FittingParams.SPEC``) and the static helpers
    that every per-parameter branch in the data / trainer / inference /
    export layers is written against.
    """

    SPEC: tuple[ParamSpec, ...] = (_FPARAM, _UPARAM, _APARAM, _CHARGE_SPIN)

    # ------------------------------------------------------------------
    # container
    # ------------------------------------------------------------------
    def __init__(self, **values: Any) -> None:
        known = {s.key for s in self.SPEC}
        unknown = set(values) - known
        if unknown:
            raise ValueError(
                f"Unknown fitting parameter(s) {sorted(unknown)}; "
                f"known: {sorted(known)}"
            )
        for spec in self.SPEC:
            setattr(self, spec.key, values.get(spec.key))

    # nice aliases matching the historical kwarg names
    @property
    def fp(self) -> Any:
        return self.fparam

    @property
    def up(self) -> Any:
        return self.uparam

    @property
    def ap(self) -> Any:
        return self.aparam

    @property
    def cs(self) -> Any:
        return self.charge_spin

    def values(self) -> dict[str, Any]:
        """Return the held tensors keyed by spec key (``None`` included)."""
        return {spec.key: getattr(self, spec.key) for spec in self.SPEC}

    def absorb(self, **current: Any) -> dict[str, Any]:
        """Fill ``current`` per-parameter values from this container.

        For every registered parameter, the explicit ``current`` value wins;
        missing ones are taken from the container.  Returns a dict keyed by
        spec key covering ALL registered parameters (``None`` for absent),
        so callers can unpack positionally:
        ``fparam, uparam, aparam, charge_spin = cond.absorb(fparam=fparam, ...)``.
        """
        vals = self.values()
        # returns a TUPLE in SPEC (ABI slot) order, so callers can unpack:
        # fparam, uparam, aparam, charge_spin = cond.absorb(fparam=fparam, ...)
        return tuple(
            current.get(k) if current.get(k) is not None else vals[k] for k in vals
        )

    @classmethod
    def from_legacy(cls, **kwargs: Any) -> "FittingParams":
        """Absorb legacy per-parameter kwargs into a container.

        ``None`` values and unknown keyword names are ignored, so callers
        can forward ``**full_kwargs`` verbatim.
        """
        known = {s.key for s in cls.SPEC}
        return cls(**{k: v for k, v in kwargs.items() if k in known and v is not None})

    # ------------------------------------------------------------------
    # spec-source helpers (static, registry-driven)
    # ------------------------------------------------------------------
    @classmethod
    def keys(cls) -> tuple[str, ...]:
        """All registered parameter keys, in ABI slot order."""
        return tuple(s.key for s in cls.SPEC)

    @classmethod
    def cond_input_keys(cls) -> tuple[str, ...]:
        """The conditioning members of the batch ``_INPUT_KEYS`` set."""
        return cls.keys()

    @classmethod
    def optional_keys(cls) -> tuple[str, ...]:
        """Keys prunable from the model inputs when ``find_<key>`` is False."""
        return tuple(s.key for s in cls.SPEC if s.optional)

    @classmethod
    def spec(cls, key: str) -> ParamSpec:
        for s in cls.SPEC:
            if s.key == key:
                return s
        raise KeyError(key)

    @staticmethod
    def _get_dim(model: Any, spec: ParamSpec) -> int:
        if spec.key == "charge_spin":
            dim = spec.call_getter(model, "get_dim_chg_spin", None)
        else:
            dim = spec.call_getter(model, spec.dim_getter, None)
        return int(dim or 0) if dim is not None else 0

    @staticmethod
    def _get_mode(model: Any, spec: ParamSpec) -> str:
        if spec.atomic_mode_getter is None:
            return "atomic" if not spec.frame_level else "frame"
        mode = spec.call_getter(model, spec.atomic_mode_getter, None)
        return str(mode) if mode is not None else "frame"

    @classmethod
    def requirement_items(cls, model: Any, requirement_item_cls: type):
        """Build ``DataRequirementItem``\\ s for every active parameter.

        Replaces the duplicated ``get_additional_data_requirement`` in the
        pt / pt_expt trainers (the two copies had already diverged on
        uparam's ``source_policy``; this is the unified behaviour).

        Parameters
        ----------
        model
            Model (or atomic model) exposing the per-parameter getters.
        requirement_item_cls
            The ``DataRequirementItem`` class to instantiate (kept a
            parameter to avoid an import cycle with ``deepmd.utils.data``).

        Returns
        -------
        list
            One item per active parameter, in registry order.
        """
        items = []
        for spec in cls.SPEC:
            dim = cls._get_dim(model, spec)
            active = (
                bool(spec.call_getter(model, spec.active_getter, False))
                if spec.active_getter is not None
                else dim > 0
            )
            if not active:
                continue
            atomic = cls._get_mode(model, spec) == "atomic"
            has_default = False
            default: Any = 0.0
            source_policy = "tracked"
            if spec.has_default_getter is not None:
                has_default = bool(spec.call_getter(model, spec.has_default_getter, False))
            if spec.default_getter is not None:
                raw = spec.call_getter(model, spec.default_getter)
                # charge_spin: the default VALUE decides (its has_default
                # getter reports the FiLM-embedding capability, not the
                # fallback presence -- mirrors the legacy chain).
                has_default = raw is not None
                if has_default:
                    default = np.asarray(raw)
            source_policy = "default" if has_default else "tracked"
            kwargs: dict[str, Any] = dict(
                must=(not has_default) if spec.optional else True,
                high_prec=False,
            )
            if spec.optional:
                kwargs.update(
                    default=default,
                    source_policy=source_policy,
                    atomic=atomic,
                )
            else:
                # aparam: always required, per-atom, no default
                kwargs.update(atomic=atomic)
            items.append(requirement_item_cls(spec.key, ndof=dim, **kwargs))
        return items

    @classmethod
    def metadata_dict(cls, model: Any) -> dict[str, Any]:
        """Exported-metadata entries for every parameter, in registry order.

        Replaces the per-parameter key block in ``_collect_metadata``; the
        charge-spin-specific ``chg_spin_table_ranges`` extra is included so
        the archive stays self-describing.
        """
        out: dict[str, Any] = {}
        for spec in cls.SPEC:
            if spec.key == "charge_spin":
                out.update(_charge_spin_metadata(model))
            else:
                out.update(_default_metadata_getter(spec, model))
        return out

    @classmethod
    def uparam_is_label(cls, model: Any) -> bool:
        """Whether ``uparam`` is a predicted LABEL (MLU) rather than input.

        True when the model's fitting predicts ``var_name == "uparam"`` and
        the model takes no uparam input (``dim_uparam == 0``) -- the MLU
        case that routes ``uparam`` from the batch into ``label_dict``.
        """
        var_name = getattr(model, "get_var_name", None)
        if var_name is not None and callable(var_name) and var_name() == "uparam":
            return cls._get_dim(model, _UPARAM) == 0
        return False

    @classmethod
    def split(
        cls,
        batch: dict[str, Any],
        model: Any = None,
        base_split: Callable[[dict[str, Any]], tuple[dict[str, Any], dict[str, Any]]] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Split a normalized batch into model inputs and labels.

        Wraps the geometry/energy split (``base_split``, the caller's
        existing implementation) and applies the registry's conditioning
        membership plus the MLU label rule.
        """
        if base_split is not None:
            input_dict, label_dict = base_split(batch)
        else:
            input_keys = {"coord", "atype", "spin", "box", "n_node", *cls.keys()}
            input_dict = {k: v for k, v in batch.items() if k in input_keys}
            label_dict = {k: v for k, v in batch.items() if k not in input_keys}
        if model is not None and cls.uparam_is_label(model) and "uparam" in input_dict:
            label_dict["uparam"] = input_dict.pop("uparam")
        return input_dict, label_dict

    @classmethod
    def prune_optionals(
        cls, input_dict: dict[str, Any], label_dict: dict[str, Any]
    ) -> dict[str, Any]:
        """Drop optional parameters whose ``find_<key>`` flag is False.

        Mutates ``input_dict`` in place and returns it (the historical
        trainer loop, registry-driven).  ``aparam`` is never pruned.
        """
        for spec in cls.SPEC:
            if not spec.optional:
                continue
            find_key = f"find_{spec.key}"
            if (
                spec.key in input_dict
                and find_key in label_dict
                and not bool(label_dict[find_key])
            ):
                input_dict.pop(spec.key)
        return input_dict

    # ------------------------------------------------------------------
    # inference-side preparation (deep_eval seam)
    # ------------------------------------------------------------------
    @classmethod
    def default_for(
        cls, key: str, metadata: dict[str, Any]
    ) -> tuple[bool, Any]:
        """Return ``(has_default, default_value)`` from exported metadata."""
        spec = cls.spec(key)
        if spec.has_default_getter is None and key != "charge_spin":
            return False, None
        has_key = f"has_default_{key}" if key != "charge_spin" else "has_default_chg_spin"
        def_key = f"default_{key}" if key != "charge_spin" else "default_chg_spin"
        if key == "charge_spin":
            # the metadata flag is ebd-based; presence of the value decides
            has_default = metadata.get(def_key) is not None
            return has_default, metadata.get(def_key)
        return bool(metadata.get(has_key, False)), metadata.get(def_key)

    @classmethod
    def prepare_lower_inputs(
        cls,
        values: dict[str, Any],
        model: Any,
        metadata: dict[str, Any],
        nframes: int,
        natoms: int,
        to_tensor: Callable[[Any, Any], Any],
    ) -> dict[str, Any]:
        """Normalize user-supplied conditioning values for a lower forward.

        Per parameter, in registry order:

        * ``dim == 0`` -> ``None`` (the ABI has no slot for it);
        * value given -> reshaped to the layout the lower expects
          (frame parameters ``(nframes, nd)``; atomic parameters flattened
          to the node axis ``(N, nd)``; atomic-mode uparam flattened
          likewise);
        * value missing -> filled from the default when one exists
          (broadcast over frames / expanded to nodes);
        * value missing and no default -> ``ValueError`` for required
          parameters, ``None`` for inactive ones.

        Parameters
        ----------
        values
            User-supplied tensors, keyed by spec key (``None`` allowed).
        model
            Model (or ``None`` when only metadata is available) probing
            ``get_dim_*`` / ``get_uparam_mode``.
        metadata
            Exported metadata dict, used for default filling in
            metadata-only inference.
        nframes, natoms
            Batch geometry.
        to_tensor
            ``lambda value, dtype_device_hint: tensor`` converter supplied
            by the caller (backend-specific).

        Returns
        -------
        dict
            ``{key: tensor | None}`` in registry order.
        """
        out: dict[str, Any] = {}
        for spec in cls.SPEC:
            key = spec.key
            dim = cls._get_dim(model, spec) if model is not None else int(
                metadata.get(f"dim_{key if key != 'charge_spin' else 'chg_spin'}", 0) or 0
            )
            if dim <= 0:
                out[key] = None
                continue
            value = values.get(key)
            atomic = cls._get_mode(model, spec) == "atomic" if model is not None else (
                str(metadata.get("uparam_mode", "frame")) == "atomic"
                if key == "uparam"
                else not spec.frame_level
            )
            if value is not None:
                out[key] = to_tensor(key, value, atomic)
                continue
            has_default, default = cls.default_for(key, metadata)
            if not has_default:
                raise ValueError(
                    f"{key} is required for this model "
                    f"(dim={dim}) but was not provided and no default is stored."
                )
            arr = np.asarray(default, dtype=np.float64).reshape(1, dim)
            if atomic:
                arr = np.broadcast_to(arr, (nframes * natoms, dim))
            else:
                arr = np.broadcast_to(arr, (nframes, dim))
            out[key] = to_tensor(key, arr, atomic)
        return out
