# SPDX-License-Identifier: LGPL-3.0-or-later
"""Unit tests for the MLU (Machine-Learning Hubbard U) fitting type."""

import numpy as np
import torch

from deepmd.dpmodel.fitting.mlu_fitting import MLUFitting as DPMLUFitting
from deepmd.pt.loss.mlu import (
    MLULoss,
)
from deepmd.pt.model.task.fitting import (
    Fitting,
)
from deepmd.pt.model.task.mlu import MLUFitting as PTMLUFitting
from deepmd.pt.utils import (
    env,
)

DEVICE = env.DEVICE


def test_mlu_registered():
    """MLU fitting type should be registered in both backends."""
    assert "mlu" in Fitting.get_plugins()


def test_mlu_frame_forward_shape():
    """Frame-mode forward returns (nf, nloc, 1) per-atom and reduced to (nf, 1)."""
    m = PTMLUFitting(ntypes=1, dim_descrpt=4, neuron=[8, 8]).to(DEVICE)
    assert m.get_intensive() is True
    desc = torch.randn(2, 4, 4, device=DEVICE)
    atype = torch.zeros(2, 4, dtype=torch.long, device=DEVICE)
    fparam = torch.tensor([[300.0], [1000.0]], device=DEVICE)
    out = m(desc, atype, fparam=fparam)
    assert "uparam" in out
    assert out["uparam"].shape == (2, 4, 1)


def test_mlu_atomic_mode_toggle():
    """set_intensive(False) switches output_def to non-intensive."""
    m = PTMLUFitting(ntypes=1, dim_descrpt=4, neuron=[8, 8]).to(DEVICE)
    assert m.output_def().get_data()["uparam"].intensive is True
    m.set_intensive(False)
    assert m.get_intensive() is False
    assert m.output_def().get_data()["uparam"].intensive is False


def test_mlu_numb_uparam_forced_zero():
    """MLU must not consume uparam as input — dim_uparam is 0."""
    m = PTMLUFitting(ntypes=1, dim_descrpt=4, neuron=[8, 8]).to(DEVICE)
    assert m.get_dim_uparam() == 0
    assert m.get_dim_fparam() == 1


def test_mlu_dpmodel_forward():
    """dpmodel-side MLU forward mirrors PT."""
    m = DPMLUFitting(ntypes=1, dim_descrpt=4, neuron=[8, 8])
    desc = np.random.randn(2, 4, 4)
    atype = np.zeros((2, 4), dtype=int)
    fparam = np.array([[300.0], [1000.0]])
    out = m(desc, atype, fparam=fparam)
    assert "uparam" in out
    assert out["uparam"].shape == (2, 4, 1)


def test_mlu_serialize_roundtrip():
    """PT serialize/deserialize preserves intensive flag and var_name."""
    for intensive in [True, False]:
        m = PTMLUFitting(
            ntypes=1, dim_descrpt=4, neuron=[8, 8], intensive=intensive
        ).to(DEVICE)
        dd = m.serialize()
        assert dd["type"] == "mlu"
        assert dd["intensive"] == intensive
        m2 = PTMLUFitting.deserialize(dd).to(DEVICE)
        assert m2.get_intensive() == intensive
        assert m2.get_task_dim() == 1


def test_mlu_loss_label_requirement():
    """MLULoss label_requirement reflects frame vs atomic mode."""
    frame_loss = MLULoss(intensive=True)
    items = frame_loss.label_requirement
    assert len(items) == 1
    assert items[0].key == "uparam"
    assert items[0].atomic is False

    atomic_loss = MLULoss(intensive=False)
    items = atomic_loss.label_requirement
    assert items[0].key == "atom_uparam"
    assert items[0].atomic is True


def test_mlu_loss_forward_frame():
    """MLULoss frame-mode forward runs without error on synthetic data."""
    from deepmd.pt.model.model import (
        get_standard_model,
    )

    params = {
        "type_map": ["Fe"],
        "descriptor": {
            "type": "se_e2_a",
            "rcut": 6.0,
            "rcut_smth": 1.8,
            "neuron": [25, 50, 100],
            "axis_neuron": 8,
            "sel": [50],
            "seed": 1,
        },
        "fitting_net": {
            "type": "mlu",
            "neuron": [120, 120, 120],
            "numb_fparam": 1,
            "intensive": True,
        },
    }
    model = get_standard_model(params).to(DEVICE)
    nf, nloc = 2, 4
    coord = torch.randn(nf, nloc, 3, device=DEVICE)
    atype = torch.zeros(nf, nloc, dtype=torch.long, device=DEVICE)
    box = torch.eye(3, device=DEVICE).unsqueeze(0).expand(nf, 3, 3).clone() * 10
    fparam = torch.tensor([[300.0], [1000.0]], device=DEVICE)
    input_dict = {"coord": coord, "atype": atype, "box": box, "fparam": fparam}
    label = {"uparam": torch.tensor([[3.0], [5.0]], device=DEVICE)}
    loss_fn = MLULoss(intensive=True, task_dim=1)
    model.atomic_model.out_std[0][0] = torch.ones(1, device=DEVICE)
    model.atomic_model.out_bias[0][0] = torch.zeros(1, device=DEVICE)
    model_pred, loss, more_loss = loss_fn(input_dict, model, label, natoms=nloc)
    assert "uparam" in model_pred
    assert loss.ndim == 0
    assert "mae" in more_loss


def test_mlu_loss_forward_atomic():
    """MLULoss atomic-mode forward runs with uparam label key (atomic shape)."""
    from deepmd.pt.model.model import (
        get_standard_model,
    )

    params = {
        "type_map": ["Fe"],
        "descriptor": {
            "type": "se_e2_a",
            "rcut": 6.0,
            "rcut_smth": 1.8,
            "neuron": [25, 50, 100],
            "axis_neuron": 8,
            "sel": [50],
            "seed": 1,
        },
        "fitting_net": {
            "type": "mlu",
            "neuron": [120, 120, 120],
            "numb_fparam": 1,
            "intensive": False,
        },
    }
    model = get_standard_model(params).to(DEVICE)
    nf, nloc = 2, 4
    coord = torch.randn(nf, nloc, 3, device=DEVICE)
    atype = torch.zeros(nf, nloc, dtype=torch.long, device=DEVICE)
    box = torch.eye(3, device=DEVICE).unsqueeze(0).expand(nf, 3, 3).clone() * 10
    fparam = torch.tensor([[300.0], [1000.0]], device=DEVICE)
    input_dict = {"coord": coord, "atype": atype, "box": box, "fparam": fparam}
    # atomic mode: label key is atom_uparam, shape (nf, nloc, 1)
    label = {"atom_uparam": torch.randn(nf, nloc, 1, device=DEVICE)}
    loss_fn = MLULoss(intensive=False, task_dim=1)
    model.atomic_model.out_std[0][0] = torch.ones(1, device=DEVICE)
    model.atomic_model.out_bias[0][0] = torch.zeros(1, device=DEVICE)
    model_pred, loss, more_loss = loss_fn(input_dict, model, label, natoms=nloc)
    assert "atom_uparam" in model_pred
    assert loss.ndim == 0
    assert "mae" in more_loss
