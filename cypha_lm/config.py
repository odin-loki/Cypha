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
    use_spectral_pde: bool = False  # FFT path; slower for small d_state sequential LM
    use_multiscale: bool = True
    use_sparse_hebbian: bool = False  # off by default — adds overhead per step

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
    context_mode: str = "full"  # full | ssm_only | gria_ngram | hybrid_gria_lstm | ...
    ngram_context: int = 2  # previous token embeds to concat (gria_ngram / ablation_no_ssm)
    train_epochs: int = 1  # multi-pass over corpus
    view_schedule: str = "same_order"  # preset name or JSON list of view transforms
    view_block_size: int = 512
    view_id_dim: int = 0  # if >0, concat view embedding into GRIA input
    view_learnable: bool = False  # online-update view table (Upgrade V2)
    view_lr: float = 0.005  # lr for view embedding updates (scaled with gria_lr_decay)
    max_view_slots: int = 16
    ngram_fusion: str = "sum"  # sum | gated (Upgrade V2 Track B)
    ngram_position_weights: bool = False
    ngram_fuse_split: bool = True  # separate field/embed projections (sum) vs single concat matmul
    gria_lr_decay: float = 0.5  # multiply gria lr each epoch after the first
    bptt_steps: int = 0  # truncated BPTT window for SSM fast weights via GRIA loss
    laplace_smoothing: float = 1.0  # unigram-style GRIA bias init / online prior
    online: bool = True
    gria_lr: float = 0.05
    ssm_lr: float = 0.001
    train_ssm: bool = False
    lstm_hidden: int = 128
    lstm_lr: float = 0.05
    hybrid_blend_logit: float = 0.0  # sigmoid -> GRIA weight in blend
    hybrid_blend_learnable: bool = True
    hybrid_blend_lr: float = 0.01
    seed: int = 42

    # Compute: auto | cpu | cuda  (env CYPHA_LM_DEVICE overrides default)
    # ``auto`` picks CPU for sequential online LM (GPU hurts small per-token ops).
    device: str = "auto"
