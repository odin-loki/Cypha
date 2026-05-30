from dataclasses import dataclass


@dataclass
class CyphaLMConfig:
    # Vocabulary
    vocab_size: int = 256
    d_embed: int = 64  # must divide ceil(log2(vocab_size))

    # CellAI SSM
    d_state: int = 128
    tau_fast: float = 1.0
    tau_slow: float = 20.0
    ssm_layers: int = 2

    # CyphaDIF Expert Field
    n_experts: int = 0
    max_experts: int = 256
    field_dim: int = 160
    nig_kappa0: float = 1.0
    nig_alpha0: float = 2.0
    nig_beta0: float = 1.0

    # GRIA Projection
    alpha_init: float = 0.5
    alpha_learnable: bool = True

    # Training
    context_length: int = 256
    online: bool = True
    gria_lr: float = 0.05
    ssm_lr: float = 0.001
    train_ssm: bool = False
    seed: int = 42
