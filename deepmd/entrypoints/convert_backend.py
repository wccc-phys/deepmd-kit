# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    Any,
)

from deepmd.backend.backend import (
    Backend,
)

log = logging.getLogger(__name__)


def _has_uparam(data: Any) -> bool:
    """Recursively check if a model dict contains uparam parameters."""
    if isinstance(data, dict):
        if data.get("numb_uparam", 0) > 0:
            return True
        if data.get("default_uparam") is not None:
            return True
        for v in data.values():
            if _has_uparam(v):
                return True
    elif isinstance(data, list):
        for item in data:
            if _has_uparam(item):
                return True
    return False


def _has_fparam(data: Any) -> bool:
    """Recursively check if a model dict contains fparam parameters."""
    if isinstance(data, dict):
        if data.get("numb_fparam", 0) > 0:
            return True
        if data.get("default_fparam") is not None:
            return True
        for v in data.values():
            if _has_fparam(v):
                return True
    elif isinstance(data, list):
        for item in data:
            if _has_fparam(item):
                return True
    return False


def convert_backend(
    *,  # Enforce keyword-only arguments
    INPUT: str,
    OUTPUT: str,
    atomic_virial: bool = False,
    **kwargs: Any,
) -> None:
    """Convert a model file from one backend to another.

    Parameters
    ----------
    INPUT : str
        The input model file.
    OUTPUT : str
        The output model file.
    atomic_virial : bool
        If True, export .pt2/.pte models with per-atom virial correction.
        This adds ~2.5x inference cost.  Default False.  Silently ignored
        (with a warning) for backends that don't support the flag.

    Notes
    -----
    Backend conversion preserves an explicit ``lower_input_kind`` reported by
    the source serializer. Sources without this metadata retain the target's
    automatic lower selection for backward compatibility. A target backend
    that cannot represent an explicit non-dense lower is rejected rather than
    silently changing the model function.
    """
    inp_backend: Backend = Backend.detect_backend_by_model(INPUT)()
    out_backend: Backend = Backend.detect_backend_by_model(OUTPUT)()
    inp_hook = inp_backend.serialize_hook
    out_hook = out_backend.deserialize_hook
    data = inp_hook(INPUT)
    # fparam/uparam models are only supported for TF <-> PT conversion
    if _has_uparam(data.get("model", {})) or _has_fparam(data.get("model", {})):
        if inp_backend.name not in ("PyTorch", "TensorFlow"):
            raise ValueError(
                f"Models with uparam or fparam from '{inp_backend.name}' backend cannot be "
                "converted. Only PyTorch and TensorFlow backends support uparam or fparam."
            )
        if out_backend.name not in ("PyTorch", "TensorFlow"):
            raise ValueError(
                f"Models with uparam or fparam cannot be converted to '{out_backend.name}' "
                "backend. Only PyTorch and TensorFlow backends support uparam or fparam."
            )
    # Forward atomic_virial to pt_expt deserialize_to_file if applicable;
    # warn and skip the flag for backends that don't accept it so that
    # scripts passing --atomic-virial indiscriminately don't break.
    import inspect

    sig = inspect.signature(out_hook)
    hook_kwargs: dict[str, Any] = {}
    lower_input_kind = data.get("lower_input_kind")
    if "lower_kind" in sig.parameters:
        hook_kwargs["lower_kind"] = (
            lower_input_kind if lower_input_kind is not None else "auto"
        )
    elif (
        lower_input_kind not in (None, "auto", "nlist")
        and not out_backend.preserves_lower_input_kind
    ):
        raise ValueError(
            f"Cannot preserve lower_input_kind {lower_input_kind!r} when "
            f"converting to output backend {out_backend.name!r}: its "
            "deserializer does not accept a lower_kind. Retrain or freeze the "
            "model with that backend instead of converting this artifact."
        )
    if "do_atomic_virial" in sig.parameters:
        hook_kwargs["do_atomic_virial"] = atomic_virial
    elif atomic_virial:
        log.warning(
            "--atomic-virial is only meaningful for pt_expt .pt2/.pte "
            "outputs; ignoring it for output backend %s",
            out_backend.name,
        )
    out_hook(OUTPUT, data, **hook_kwargs)
