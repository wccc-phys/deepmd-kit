# SPDX-License-Identifier: LGPL-3.0-or-later
"""Loss for the MLU (Machine-Learning Hubbard U) fitting type.

MLU predicts ``uparam`` — the Hubbard U value — from the descriptor and an
external frame parameter (typically temperature). Two output modes:

- **frame mode** (``intensive=True``): a single per-frame U. The dataset label
  ``uparam.npy`` has shape ``(N, 1)``. The model's reduced per-frame output is
  compared directly against the label.
- **atomic mode** (``intensive=False``): one U per atom per frame. The dataset
  label has shape ``(N, natoms)``. The model's per-atom output
  (``atom_uparam``) is compared against the per-atom label.
"""

import logging
from typing import (
    Any,
)

import torch
import torch.nn.functional as F

from deepmd.pt.loss.loss import (
    TaskLoss,
)
from deepmd.pt.utils import (
    env,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.version import (
    check_version_compatibility,
)

log = logging.getLogger(__name__)


@TaskLoss.register("mlu")
class MLULoss(TaskLoss):
    """Loss for MLU fitting (predicts Hubbard U)."""

    def __init__(
        self,
        task_dim: int = 1,
        intensive: bool = True,
        loss_func: str = "smooth_mae",
        metric: list[str] | None = None,
        beta: float = 1.00,
        out_bias: list | None = None,
        out_std: list | None = None,
        **kwargs: Any,
    ) -> None:
        r"""Construct a layer to compute loss on the Hubbard U.

        Parameters
        ----------
        task_dim : int
            The output dimension (always 1 for MLU).
        intensive : bool
            Whether the U is intensive (per-frame, mean-reduced). ``True`` =
            frame mode, ``False`` = atomic mode.
        loss_func : str
            One of "smooth_mae", "mae", "rmse", "mse", "mape".
        metric : list[str]
            Metrics printed to the lcurve file.
        beta : float
            ``beta`` parameter in ``smooth_mae``.
        out_bias, out_std : list | None
            Optional per-type output bias / std. If None, read from the model
            atomic_model (computed by ``compute_output_stats``).
        """
        super().__init__()
        if metric is None:
            metric = ["mae"]
        self.task_dim = task_dim
        self.intensive = intensive
        self.loss_func = loss_func
        self.metric = metric
        self.beta = beta
        self.out_bias = out_bias
        self.out_std = out_std
        self.var_name = "uparam"

    def forward(
        self,
        input_dict: dict[str, torch.Tensor],
        model: torch.nn.Module,
        label: dict[str, torch.Tensor],
        natoms: int,
        learning_rate: float = 0.0,
        mae: bool = False,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, dict[str, torch.Tensor]]:
        """Return loss on Hubbard U.

        For frame mode: compares the per-frame reduced U against the
        ``(N, 1)`` label.

        For atomic mode: compares the per-atom U against the per-atom label
        ``(N, natoms, 1)``.
        """
        model_pred = model(**input_dict)
        var_name = self.var_name

        if self.intensive:
            # frame mode: per-frame reduced output, shape (nbz, task_dim)
            pred = model_pred[var_name]
            target = label[var_name]
            assert pred.shape == target.shape, (
                f"MLU frame mode shape mismatch: pred {pred.shape} vs label {target.shape}"
            )
        else:
            # atomic mode: per-atom output, shape (nbz, nloc, task_dim)
            atom_key = f"atom_{var_name}"
            pred = model_pred[atom_key]
            target = label[atom_key]
            assert pred.shape == target.shape, (
                f"MLU atomic mode shape mismatch: pred {pred.shape} vs label {target.shape}"
            )

        # out_std / out_bias come from compute_output_stats on the model.
        if self.out_std is None:
            out_std = model.atomic_model.out_std[0][0]
        else:
            out_std = torch.tensor(
                self.out_std, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE
            )
        if self.out_bias is None:
            out_bias = model.atomic_model.out_bias[0][0]
        else:
            out_bias = torch.tensor(
                self.out_bias, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE
            )

        loss = torch.zeros(1, dtype=env.GLOBAL_PT_FLOAT_PRECISION, device=env.DEVICE)[0]
        more_loss: dict[str, torch.Tensor] = {}

        norm_pred = (pred - out_bias) / out_std
        norm_target = (target - out_bias) / out_std
        if self.loss_func == "smooth_mae":
            loss += F.smooth_l1_loss(
                norm_target, norm_pred, reduction="sum", beta=self.beta
            )
        elif self.loss_func == "mae":
            loss += F.l1_loss(norm_target, norm_pred, reduction="sum")
        elif self.loss_func == "mse":
            loss += F.mse_loss(norm_target, norm_pred, reduction="sum")
        elif self.loss_func == "rmse":
            loss += torch.sqrt(F.mse_loss(norm_target, norm_pred, reduction="mean"))
        elif self.loss_func == "mape":
            loss += torch.mean(torch.abs((target - pred) / (target + 1e-3)))
        else:
            raise RuntimeError(f"Unknown loss function : {self.loss_func}")

        if "smooth_mae" in self.metric:
            more_loss["smooth_mae"] = F.smooth_l1_loss(
                target, pred, reduction="mean", beta=self.beta
            ).detach()
        if "mae" in self.metric:
            more_loss["mae"] = F.l1_loss(target, pred, reduction="mean").detach()
        if "mse" in self.metric:
            more_loss["mse"] = F.mse_loss(target, pred, reduction="mean").detach()
        if "rmse" in self.metric:
            more_loss["rmse"] = torch.sqrt(
                F.mse_loss(target, pred, reduction="mean")
            ).detach()
        if "mape" in self.metric:
            more_loss["mape"] = torch.mean(
                torch.abs((target - pred) / (target + 1e-3))
            ).detach()

        return model_pred, loss, more_loss

    @property
    def label_requirement(self) -> list[DataRequirementItem]:
        """Return data label requirements needed for this loss calculation.

        Frame mode: key ``uparam``, file ``uparam.npy`` shape ``(N, 1)``,
        ``atomic=False``.

        Atomic mode: key ``atom_uparam``, file ``uparam.npy`` shape
        ``(N, natoms)`` (loaded via ``atom_uparam.npy`` → ``uparam.npy``
        fallback in ``_get_data_path``), ``atomic=True``.
        """
        key = self.var_name if self.intensive else f"atom_{self.var_name}"
        return [
            DataRequirementItem(
                key,
                ndof=self.task_dim,
                atomic=not self.intensive,
                must=True,
                high_prec=True,
            )
        ]

    def serialize(self) -> dict:
        return {
            "@class": "MLULoss",
            "@version": 1,
            "task_dim": self.task_dim,
            "intensive": self.intensive,
            "loss_func": self.loss_func,
            "metric": self.metric,
            "beta": self.beta,
            "out_bias": self.out_bias,
            "out_std": self.out_std,
        }

    @classmethod
    def deserialize(cls, data: dict) -> "MLULoss":
        data = data.copy()
        check_version_compatibility(data.pop("@version"), 1, 1)
        data.pop("@class")
        return cls(**data)
