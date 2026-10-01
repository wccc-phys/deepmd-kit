# SPDX-License-Identifier: LGPL-3.0-or-later
from typing import (
    Any,
)

from deepmd.dpmodel.fitting.mlu_fitting import (
    MLUFitting as MLUFittingDP,
)
from deepmd.pt_expt.common import (
    torch_module,
)
from deepmd.pt_expt.fitting.base_fitting import (
    BaseFitting,
)


@BaseFitting.register("mlu")
@torch_module
class MLUFitting(MLUFittingDP):
    def share_params(
        self,
        base_class: Any,
        shared_level: int,
        model_prob: float = 1.0,
        protection: float = 1e-2,
        resume: bool = False,
    ) -> None:
        """Share parameters with base_class for multi-task training.

        Delegates to the shared InvarFitting implementation: the MLU fitting
        is an InvarFitting with ``var_name="uparam"`` and the same fparam /
        aparam / uparam statistics buffers.
        """
        from deepmd.pt_expt.fitting.invar_fitting import (
            InvarFitting,
        )

        return InvarFitting.share_params(
            self, base_class, shared_level, model_prob, protection, resume
        )
