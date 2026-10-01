# SPDX-License-Identifier: LGPL-3.0-or-later
import logging
from typing import (
    Any,
)

import torch

from deepmd.dpmodel import (
    FittingOutputDef,
    OutputVariableDef,
)
from deepmd.pt.model.task.ener import (
    InvarFitting,
)
from deepmd.pt.model.task.fitting import (
    Fitting,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.pt.utils.env import (
    DEFAULT_PRECISION,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

dtype = env.GLOBAL_PT_FLOAT_PRECISION
device = env.DEVICE

log = logging.getLogger(__name__)


@Fitting.register("mlu")
class MLUFitting(InvarFitting):
    """Fitting the Hubbard U value (uparam) of the system.

    Predicts ``uparam`` (the DFT+U Hubbard U) from the descriptor and an
    external frame parameter (typically temperature, passed via ``fparam``).
    Unlike the ``ener`` fitting, ``uparam`` here is the *output*, not an input
    — the uparam-as-input concat pipeline in :class:`GeneralFitting` is
    disabled by forcing ``numb_uparam=0``.

    Two output modes are selected by ``intensive``:

    - ``intensive=True`` (frame mode): per-atom outputs are averaged into a
      single per-frame U. Matches dataset ``uparam.npy`` of shape ``(N, 1)``.
    - ``intensive=False`` (atomic mode): per-atom outputs are summed. The
      per-atom contribution equals ``U_atom / natoms`` after training, so the
      summed per-frame value equals the total U; the per-atom output is
      recovered by ``apply_out_stat`` and the atomic_output path. Matches
      dataset ``uparam.npy`` of shape ``(N, natoms)``.

    Parameters
    ----------
    ntypes : int
        Element count.
    dim_descrpt : int
        Embedding width per atom.
    neuron : list[int]
        Number of neurons in each hidden layer of the fitting net.
    intensive : bool, optional
        Whether the U is intensive (per-frame, mean-reduced). ``None`` means
        auto-detect from dataset ``uparam.npy`` shape at training start.
    numb_fparam : int
        Number of frame parameters (typically 1 for temperature).
    numb_aparam : int
        Number of atomic parameters.
    dim_case_embd : int
        Dimension of case specific embedding.
    bias_atom_p : torch.Tensor, optional
        Average property per atom for each element.
    resnet_dt : bool
        Using time-step in the ResNet construction.
    activation_function : str
        Activation function.
    precision : str
        Numerical precision.
    mixed_types : bool
        If true, use a uniform fitting net for all atom types.
    seed : int, optional
        Random seed.
    """

    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        neuron: list[int] = [128, 128, 128],
        bias_atom_p: torch.Tensor | None = None,
        intensive: bool | None = None,
        resnet_dt: bool = True,
        numb_fparam: int = 1,
        numb_aparam: int = 0,
        dim_case_embd: int = 0,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        mixed_types: bool = True,
        trainable: bool | list[bool] = True,
        seed: int | None = None,
        default_fparam: list | None = None,
        distinguish_types: bool = True,
        **kwargs: Any,
    ) -> None:
        self.task_dim = 1
        # intensive is None means "auto-detect from dataset" — resolved in
        # the trainer after inspecting uparam.npy. Default to True so the
        # output_def is well-formed before detection runs.
        self.intensive = intensive if intensive is not None else True
        self._intensive_explicit = intensive is not None
        self.distinguish_types = distinguish_types
        super().__init__(
            var_name="uparam",
            ntypes=ntypes,
            dim_descrpt=dim_descrpt,
            dim_out=1,
            neuron=neuron,
            bias_atom_e=bias_atom_p,
            resnet_dt=resnet_dt,
            # fparam (temperature) is the input; uparam is the OUTPUT, so
            # force numb_uparam=0 to skip the uparam-as-input concat pipeline.
            numb_fparam=numb_fparam,
            numb_uparam=0,
            numb_aparam=numb_aparam,
            dim_case_embd=dim_case_embd,
            activation_function=activation_function,
            precision=precision,
            mixed_types=mixed_types,
            trainable=trainable,
            seed=seed,
            default_fparam=default_fparam,
            default_uparam=None,
            **kwargs,
        )

    def output_def(self) -> FittingOutputDef:
        return FittingOutputDef(
            [
                OutputVariableDef(
                    self.var_name,
                    [self.dim_out],
                    reducible=True,
                    r_differentiable=False,
                    c_differentiable=False,
                    intensive=self.intensive,
                ),
            ]
        )

    def get_intensive(self) -> bool:
        """Whether the U is intensive (per-frame, mean-reduced)."""
        return self.intensive

    def get_task_dim(self) -> int:
        return self.task_dim

    def get_distinguish_types(self) -> bool:
        """Get whether to distinguish atom types when computing output stats."""
        return self.distinguish_types

    def set_intensive(self, intensive: bool) -> None:
        """Update intensive after auto-detection. Rebuilds output_def."""
        self.intensive = bool(intensive)
        self._intensive_explicit = True

    @classmethod
    def deserialize(cls, data: dict) -> "MLUFitting":
        data = data.copy()
        check_version_compatibility(data.pop("@version", 1), 1, 1)
        data.setdefault("distinguish_types", False)
        data.pop("dim_out")
        data.pop("var_name", None)
        # GeneralFitting.serialize writes these; MLUFitting.__init__ sets
        # numb_uparam=0 and default_uparam=None unconditionally (uparam is
        # the OUTPUT of MLU), so pop them to avoid a duplicate-keyword error.
        data.pop("numb_uparam", None)
        data.pop("default_uparam", None)
        # InvarFitting.serialize emits these; they are unused by MLUFitting.
        for k in (
            "atom_ener",
            "tot_ener_zero",
            "layer_name",
            "use_aparam_as_mask",
            "spin",
        ):
            data.pop(k, None)
        obj = super().deserialize(data)
        return obj

    def serialize(self) -> dict:
        """Serialize the fitting to dict."""
        dd = {
            **InvarFitting.serialize(self),
            "type": "mlu",
            "task_dim": self.task_dim,
            "intensive": self.intensive,
            "distinguish_types": self.distinguish_types,
        }
        # var_name is fixed to "uparam" for MLU — drop it from the serialized
        # payload so deserialize doesn't try to pass it as a ctor argument.
        dd.pop("var_name", None)
        dd["@version"] = 1
        return dd

    # make jit happy with torch 2.0.0
    exclude_types: list[int]
