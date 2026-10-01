# SPDX-License-Identifier: LGPL-3.0-or-later

from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.common import (
    DEFAULT_PRECISION,
)
from deepmd.dpmodel.fitting.invar_fitting import (
    InvarFitting,
)
from deepmd.dpmodel.output_def import (
    FittingOutputDef,
    OutputVariableDef,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


@InvarFitting.register("mlu")
class MLUFitting(InvarFitting):
    r"""Fitting the Hubbard U value (uparam) of the system.

    Predicts ``uparam`` (the DFT+U Hubbard U) from the descriptor and an
    external frame parameter (typically temperature, passed via ``fparam``).
    Unlike the ``ener`` fitting, ``uparam`` here is the *output*, not an input
    — the uparam-as-input concat pipeline in :class:`GeneralFitting` is
    disabled by forcing ``numb_uparam=0``.

    Two output modes are selected by ``intensive``:

    - ``intensive=True`` (frame mode): per-atom outputs are averaged into a
      single per-frame U. Matches dataset ``uparam.npy`` of shape ``(N, 1)``.
    - ``intensive=False`` (atomic mode): per-atom outputs are summed. Matches
      dataset ``uparam.npy`` of shape ``(N, natoms)``.

    Parameters
    ----------
    ntypes
            The number of atom types.
    dim_descrpt
            The dimension of the input descriptor.
    neuron
            Number of neurons :math:`N` in each hidden layer of the fitting net
    bias_atom_p
            Average property per atom for each element.
    rcond
            The condition number for the regression of atomic energy.
    trainable
            If the weights of fitting net are trainable.
    intensive
            Whether the U is intensive (per-frame). ``None`` means auto-detect
            from dataset ``uparam.npy`` shape at training start.
    resnet_dt
            Time-step `dt` in the resnet construction.
    numb_fparam
            Number of frame parameter (typically 1 for temperature).
    numb_aparam
            Number of atomic parameter.
    activation_function
            The activation function :math:`\boldsymbol{\phi}` in the embedding net.
    precision
            The precision of the embedding net parameters.
    mixed_types
            If false, different atomic types uses different fitting net, otherwise
            different atom types share the same fitting net.
    exclude_types: list[int]
            Atomic contributions of the excluded atom types are set zero.
    type_map: list[str], Optional
            A list of strings. Give the name to each type of atoms.
    default_fparam: list[float], optional
        The default frame parameter.
    distinguish_types : bool
            Whether to distinguish atom types when computing output statistics.
    """

    def __init__(
        self,
        ntypes: int,
        dim_descrpt: int,
        neuron: list[int] = [128, 128, 128],
        bias_atom_p: Array | None = None,
        rcond: float | None = None,
        trainable: bool | list[bool] = True,
        intensive: bool | None = None,
        task_dim: int = 1,
        resnet_dt: bool = True,
        numb_fparam: int = 1,
        numb_aparam: int = 0,
        dim_case_embd: int = 0,
        activation_function: str = "tanh",
        precision: str = DEFAULT_PRECISION,
        mixed_types: bool = True,
        exclude_types: list[int] = [],
        type_map: list[str] | None = None,
        default_fparam: list | None = None,
        distinguish_types: bool = True,
        # not used
        seed: int | None = None,
    ) -> None:
        self.task_dim = task_dim
        self.intensive = intensive if intensive is not None else True
        self._intensive_explicit = intensive is not None
        self.distinguish_types = distinguish_types
        super().__init__(
            var_name="uparam",
            ntypes=ntypes,
            dim_descrpt=dim_descrpt,
            dim_out=1,
            neuron=neuron,
            bias_atom=bias_atom_p,
            resnet_dt=resnet_dt,
            # fparam (temperature) is the input; uparam is the OUTPUT.
            numb_fparam=numb_fparam,
            numb_uparam=0,
            numb_aparam=numb_aparam,
            dim_case_embd=dim_case_embd,
            rcond=rcond,
            trainable=trainable,
            activation_function=activation_function,
            precision=precision,
            mixed_types=mixed_types,
            exclude_types=exclude_types,
            type_map=type_map,
            default_fparam=default_fparam,
            default_uparam=None,
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
                *self._middle_output_def(),
            ]
        )

    def get_intensive(self) -> bool:
        """Whether the U is intensive (per-frame, mean-reduced)."""
        return self.intensive

    def get_task_dim(self) -> int:
        return self.task_dim

    def get_distinguish_types(self) -> bool:
        """Get whether the fitting net computes stats which are distinguished between different types of atoms."""
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
        dd.pop("var_name", None)
        dd["@version"] = 1
        return dd
