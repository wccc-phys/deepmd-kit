# SPDX-License-Identifier: LGPL-3.0-or-later
"""Composite DP+U inference wrapper.

Loads a DP+U model (a :class:`DeepPot` whose fitting accepts ``uparam`` as an
input) and an MLU model (a :class:`DeepProperty` with ``var_name="uparam"``
that *predicts* U). At inference time, MLU predicts U from the configuration
and ``fparam`` (typically temperature); the predicted U is then fed into the
DP+U model to obtain energy/force/virial.
"""

from __future__ import (
    annotations,
)

import logging
from typing import (
    Any,
)

import numpy as np

from .deep_eval import (
    DeepEval,
)

log = logging.getLogger(__name__)

__all__ = ["DeepDpU"]


class DeepDpU:
    """Composite DP+U + MLU evaluator.

    Parameters
    ----------
    dp_model : str | Path
        Path to the frozen DP+U model file (must accept ``uparam`` input).
    mlu_model : str | Path
        Path to the frozen MLU model file (predicts ``uparam``).
    dp_head : str, optional
        Head name for the DP+U model, forwarded to :class:`DeepEval`.
    mlu_head : str, optional
        Head name for the MLU model, forwarded to :class:`DeepEval`.
    auto_batch_size : bool | int, default: True
        Auto batch size for both sub-models.
    neighbor_list : optional
        ASE neighbor list class. Forwarded to both sub-models.
    **kwargs : dict
        Other keyword arguments forwarded to :class:`DeepEval` for both
        sub-models.
    """

    def __init__(
        self,
        dp_model: str,
        mlu_model: str,
        dp_head: str | None = None,
        mlu_head: str | None = None,
        auto_batch_size: bool | int = True,
        neighbor_list: Any | None = None,
        **kwargs: Any,
    ) -> None:
        self.dp = DeepEval(
            dp_model,
            head=dp_head,
            auto_batch_size=auto_batch_size,
            neighbor_list=neighbor_list,
            **kwargs,
        )
        self.mlu = DeepEval(
            mlu_model,
            head=mlu_head,
            auto_batch_size=auto_batch_size,
            neighbor_list=neighbor_list,
            **kwargs,
        )
        if self.dp.get_dim_uparam() < 1:
            raise ValueError(
                "DP+U model does not take uparam input (dim_uparam=0). "
                "DeepDpU requires a DP+U model that consumes uparam."
            )
        mlu_var = self.mlu.get_var_name()
        if mlu_var != "uparam":
            raise ValueError(
                f"MLU model var_name is '{mlu_var}', expected 'uparam'. "
                "The MLU model must have been trained with fitting_net.type='mlu'."
            )
        if self.mlu.get_task_dim() != 1:
            raise ValueError(f"MLU task_dim is {self.mlu.get_task_dim()}, expected 1.")

    def eval(
        self,
        coords: np.ndarray,
        cells: np.ndarray | None,
        atom_types: list[int] | np.ndarray,
        atomic: bool = False,
        fparam: np.ndarray | None = None,
        aparam: np.ndarray | None = None,
        mixed_type: bool = False,
        **kwargs: Any,
    ) -> tuple[np.ndarray, ...]:
        """Evaluate energy/force/virial, with U predicted by the MLU sub-model.

        Arguments mirror :meth:`DeepPot.eval`, except ``uparam`` is not
        accepted — it is predicted internally by the MLU model.

        Returns
        -------
        tuple
            Same as :meth:`DeepPot.eval`: ``(energy, force, virial[, atomic_energy, atomic_virial])``.
            Use ``atomic=True`` to get the atomic-energy / atomic-virial extensions.
        """
        (uparam,) = self.mlu.eval(
            coords,
            cells,
            atom_types,
            atomic=False,
            fparam=fparam,
            aparam=aparam,
            mixed_type=mixed_type,
        )
        # uparam shape: (nframes, 1) for frame-mode MLU.
        return self.dp.eval(
            coords,
            cells,
            atom_types,
            atomic=atomic,
            fparam=fparam,
            aparam=aparam,
            uparam=uparam,
            mixed_type=mixed_type,
            **kwargs,
        )

    def get_dim_fparam(self) -> int:
        """Get the dimension of the frame parameter (input to both sub-models)."""
        return self.dp.get_dim_fparam()

    def get_dim_uparam(self) -> int:
        """Get the dimension of the U parameter that the MLU sub-model predicts."""
        return self.mlu.get_task_dim()

    def get_ntypes(self) -> int:
        """Get the number of atom types supported by both sub-models."""
        if self.dp.get_ntypes() != self.mlu.get_ntypes():
            log.warning(
                "DP+U and MLU sub-models have different ntypes: %d vs %d",
                self.dp.get_ntypes(),
                self.mlu.get_ntypes(),
            )
        return self.dp.get_ntypes()
