import torch
import torch.nn as nn


def make_encoder_optimizer(
    full_model: nn.Module,
    lr: float,
    weight_decay: float = 0.0,
) -> torch.optim.Adam:
    """Adam with weight decay on encoder linear weights only.

    Expects full_model = nn.Sequential(user_encoder, FactorLayer) as
    produced by StructuralModel.build().  weight_decay is applied to
    parameters whose name ends with '.weight' and does not contain 'bn';
    all other params (biases, BN, FactorLayer) receive weight_decay=0.

    Setting weight_decay=0.02 approximates Keras kernel_regularizer=l1_l2(l2=0.01)
    since PyTorch weight_decay = 2 * l2.  L1 cannot be replicated via weight_decay.
    """
    encoder = full_model[0]
    factor = full_model[1]
    lin_w = [
        p
        for n, p in encoder.named_parameters()
        if n.endswith(".weight") and "bn" not in n
    ]
    other = [
        p
        for n, p in encoder.named_parameters()
        if not (n.endswith(".weight") and "bn" not in n)
    ]
    return torch.optim.Adam(
        [
            {"params": lin_w, "weight_decay": weight_decay},
            {"params": other, "weight_decay": 0.0},
            {"params": list(factor.parameters()), "weight_decay": 0.0},
        ],
        lr=lr,
    )
