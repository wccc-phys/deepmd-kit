# SPDX-License-Identifier: LGPL-3.0-or-later
"""Loss for MLU (Machine-Learning Hubbard U) — dpmodel backend.

See ``deepmd.pt.loss.mlu`` for design notes. The atomic-mode label has shape
``(N, natoms, 1)``; the model produces a per-atom prediction under the key
``atom_uparam``. Frame-mode label has shape ``(N, 1)``; the model produces a
per-frame prediction under ``uparam``.
"""

from typing import (
    Any,
)

import array_api_compat

from deepmd.dpmodel.array_api import (
    Array,
)
from deepmd.dpmodel.loss.loss import (
    Loss,
)
from deepmd.utils.data import (
    DataRequirementItem,
)
from deepmd.utils.version import (
    check_version_compatibility,
)


class MLULoss(Loss):
    r"""Loss on MLU (Hubbard U) predictions.

    Parameters
    ----------
    task_dim : int
        Output dimension (always 1 for MLU).
    intensive : bool
        ``True`` = frame mode (per-frame U), ``False`` = atomic mode
        (per-atom U).
    loss_func : str
        "smooth_mae", "mae", "mse", "rmse", "mape".
    metric : list[str]
        Metrics to report.
    beta : float
        ``beta`` parameter in ``smooth_mae``.
    out_bias, out_std : list | None
        Optional per-type output bias / std.
    """

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
        if metric is None:
            metric = ["mae"]
        self.task_dim = task_dim
        self.intensive = intensive
        self.var_name = "uparam"
        self.loss_func = loss_func
        self.metric = metric
        self.beta = beta
        self.out_bias = out_bias
        self.out_std = out_std

    def call(
        self,
        learning_rate: float,
        natoms: int,
        model_dict: dict[str, Array],
        label_dict: dict[str, Array],
        mae: bool = False,
    ) -> tuple[Array, dict[str, Array]]:
        """Calculate loss from model results and labeled results."""
        del learning_rate, mae
        var_name = self.var_name

        if self.intensive:
            pred = model_dict[var_name]
            label = label_dict[var_name]
        else:
            pred = model_dict[f"atom_{var_name}"]
            label = label_dict[f"atom_{var_name}"]

        xp = array_api_compat.array_namespace(pred)
        dev = array_api_compat.device(pred)

        if self.out_std is not None:
            out_std = xp.asarray(self.out_std, dtype=pred.dtype, device=dev)
        else:
            out_std = xp.ones((self.task_dim,), dtype=pred.dtype, device=dev)
        if self.out_bias is not None:
            out_bias = xp.asarray(self.out_bias, dtype=pred.dtype, device=dev)
        else:
            out_bias = xp.zeros((self.task_dim,), dtype=pred.dtype, device=dev)

        loss = xp.zeros((), dtype=pred.dtype, device=dev)
        more_loss = {}

        norm_pred = (pred - out_bias) / out_std
        norm_label = (label - out_bias) / out_std
        diff = norm_label - norm_pred

        if self.loss_func == "smooth_mae":
            abs_diff = xp.abs(diff)
            smooth_l1 = xp.where(
                abs_diff < self.beta,
                0.5 * diff**2 / self.beta,
                abs_diff - 0.5 * self.beta,
            )
            loss = loss + xp.sum(smooth_l1)
        elif self.loss_func == "mae":
            loss = loss + xp.sum(xp.abs(diff))
        elif self.loss_func == "mse":
            loss = loss + xp.sum(xp.square(diff))
        elif self.loss_func == "rmse":
            loss = loss + xp.sqrt(xp.mean(xp.square(diff)))
        elif self.loss_func == "mape":
            loss = loss + xp.mean(xp.abs((label - pred) / (label + 1e-3)))
        else:
            raise RuntimeError(f"Unknown loss function : {self.loss_func}")

        if "smooth_mae" in self.metric:
            abs_raw = xp.abs(label - pred)
            more_loss["smooth_mae"] = xp.mean(
                xp.where(
                    abs_raw < self.beta,
                    0.5 * (label - pred) ** 2 / self.beta,
                    abs_raw - 0.5 * self.beta,
                )
            )
        if "mae" in self.metric:
            more_loss["mae"] = xp.mean(xp.abs(label - pred))
        if "mse" in self.metric:
            more_loss["mse"] = xp.mean(xp.square(label - pred))
        if "rmse" in self.metric:
            more_loss["rmse"] = xp.sqrt(xp.mean(xp.square(label - pred)))
        if "mape" in self.metric:
            more_loss["mape"] = xp.mean(xp.abs((label - pred) / (label + 1e-3)))

        return loss, more_loss

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
