import yaml
import os


class SamplingConfig:
    """Sampling configuration for generation."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        fields = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        for k in self.__class__.__annotations__:
            if k not in fields:
                fields[k] = getattr(self, k, None)
        items = ", ".join(f"{k}={v!r}" for k, v in fields.items())
        return f"SamplingConfig({items})"

    sampling_method: str = "ode"
    num_sampling_steps: list = [50]
    cfgs: list = [1]
    self_cond_cfg_scales: list = [1.0]
    time_schedule: str = "logit_normal"  # 'logit_normal' or 'uniform'
    sde_gamma: float = 0.0  # Per-step SDE churn fraction; 0.0 -> pure ODE. Used when sampling_method == "sde".


# ============================================
# Configuration
# ============================================
class Config:
    # Dataset
    data_path: str = None
    eval_data_path: str = None
    max_length: int = 128
    max_input_length: int = None  # Max length for conditioning input (e.g., prompt or encoder input); None = no limit
    pad_token: str = "pad"  # "pad" or "eos" - which token to use for padding
    code_target: bool = False  # Targets are Python escaped by utils.code_utils (TinyGSM); score samples by execution
    exec_num_proc: int = 16  # Programs executed in parallel when code_target is set
    exec_timeout: float = 10.0  # Wall-clock seconds allowed per generated program

    # Tokenizer
    tokenizer_name: str = None  # Defaults to encoder_model_name if not set

    # Encoder
    encoder_model_name: str = "t5-small"
    encoder_checkpoint: str = None
    latent_mean: float = 0.0
    latent_std: float = 1.0

    # Model architecture
    model: str = "ELF-B"
    bottleneck_dim: int = 128  # Bottleneck dimension for text projection
    num_time_tokens: int = 4  # Number of in-context time conditioning tokens
    num_self_cond_cfg_tokens: int = 4  # Number of in-context self-cond CFG tokens
    num_model_mode_tokens: int = 4  # If > 0, prepend learnable model-mode tokens that signal decoding mode
    attn_dropout: float = 0.0
    proj_dropout: float = 0.0

    # Denoiser objective
    denoiser_p_mean: float = 0.8
    denoiser_p_std: float = 0.8
    denoiser_noise_scale: float = 1.0
    t_eps: float = 5e-2
    time_schedule: str = "logit_normal"  # 'logit_normal' or 'uniform'

    # CorrNetwork correction strength at generation time (opt-in diagnostic, off by
    # default). v_con = v_ind + corr_strength * phi_eta(...). 1.0 matches training
    # exactly; sweeping toward 0.0 isolates whether phi_eta's correction itself (as
    # opposed to the sampling schedule / sde_gamma) is responsible for reduced
    # generation diversity relative to the frozen Stage 1 backbone.
    corr_strength: float = 1.0

    # Context-order ablation: "causal" (default) keeps C_i = the gp_q tokens
    # immediately preceding i in natural sequence order, matching training's
    # original design and every published result. "random" replaces this with
    # a per-sample random permutation of the sequence (same permutation for
    # the GP-path target and the CorrNetwork's causal conv, fixed for a whole
    # generation trajectory at inference) -- see sample_random_order and
    # make_corr_model_apply_fn in utils/sampling_utils.py. Only the noise-GP
    # path (plain sample_gp_path, i.e. use_path_gp/use_clayton_gp/use_pde_gp/
    # use_cnp_gp all False) implements this; other GP variants are unaffected.
    corr_context_order: str = 'causal'

    # GP path (Stage 2: correlation-aware path)
    gp_q: int = 2             # Context window size: token i conditions on the previous gp_q tokens
    gp_rho: float = 0.5       # Adjacent-token correlation (noise-GP kernel denominator).
    gp_kernel: str = 'rbf'    # Kernel: 'rbf', 'exponential', or 'content'.
    # Path-GP fields (use_path_gp=True switches from noise-GP to deterministic path-GP)
    use_path_gp: bool = False
    gp_path_strength: float = 1.0   # Blend strength γ ∈ [0, 1]
    gp_lengthscale: float = 1.0     # Kernel length scale ℓ for path-GP
    gp_obs_noise: float = 1.0       # GP observation noise; set to ~observation variance (ε*σ has var~4)
    gp_jitter: float = 1e-5         # Numerical jitter (all diagonal entries)

    # Clayton copula path — JAX training (use_clayton_gp=True)
    # Markov (first-order) causal chain; gp_q is accepted but unused (parity only).
    use_clayton_gp: bool = False
    clayton_theta: float = 1.0      # Clayton dependence parameter (> 0; larger = stronger lower-tail dependence)

    # CNP path — JAX training (use_cnp_gp=True). Trainable alternative to the fixed
    # GP/Clayton paths: a small causal network predicts per-token correlation
    # strength from the real embeddings' own local predictability, trained jointly
    # with CorrNetwork via an auxiliary NLL loss (weighted by cnp_loss_weight).
    use_cnp_gp: bool = False
    cnp_model: str = 'CNPPathNet-S'  # CNPPathNet-S (h=128), CNPPathNet-M (h=256)
    cnp_checkpoint: str = None       # path to a pretrain_cnp.py output dir (frozen, phase-1 pretrained)
    cnp_loss_weight: float = 1.0     # unused in the two-phase (frozen-CNP) design; kept for parity

    # PDE-inspired GP path — JAX training (use_pde_gp=True)
    use_pde_gp: bool = False
    pde_time_scale: float = 1.0         # τ: time-dimension weight in GP distance
    pde_context_scale: float = 1.0      # κ: context-path weight in GP distance
    pde_log_length_scale: float = 0.0   # log ℓ (clamped to [-5, 3])
    pde_denominator_eps: float = 1e-6   # fallback threshold for 1−a²

    # PDE-inspired GP path (PyTorch, experimental; path_type: pde_inspired_gp)
    # Example YAML:
    #   path_type: pde_inspired_gp
    #   pde_inspired_gp:
    #     q: 2
    #     log_length_scale: 0.0
    #     trainable_length_scale: false
    #     time_scale: 1.0
    #     context_scale: 1.0
    #     denominator_eps: 1.0e-6
    path_type: str = None
    pde_inspired_gp: dict = None

    # CorrNetwork φ_η (Stage 2: correlation network)
    corr_model: str = 'CorrNet-M'  # Size variant: CorrNet-S (h=128), CorrNet-M (h=256), CorrNet-L (h=512)
    backbone_checkpoint: str = None  # Path to frozen Stage 1 checkpoint (Stage 2 only)

    # Stage 1 (train.py) ablation: fine-tune a PRETRAINED backbone with the
    # plain linear-OT objective instead of training from random init.
    # backbone_checkpoint doubles as the weights source here (same field
    # Stage 2 uses); when this is True, train.py loads ONLY params/ema_params
    # from it via train_stage2.load_frozen_backbone (handles both this
    # repo's msgpack format and HF's orbax ocdbt format) and starts the
    # optimizer/step counter fresh at 0 -- NOT a --resume (which would also
    # carry over opt_state/step/epoch from the checkpoint and make max_steps
    # a no-op if the checkpoint's own step already exceeds it).
    init_from_backbone_checkpoint: bool = False

    # Stage 1 (train.py) ablation: freeze the pretrained backbone entirely and
    # ONLY train LoRA low-rank adapters (A: (in,rank) small-random, B: (rank,out)
    # zero-init) injected into every block's attn qkv/proj and mlp w12/w3 --
    # see modules/layers.py:lora_delta. Requires init_from_backbone_checkpoint
    # also set (LoRA fine-tunes a pretrained checkpoint, not random init).
    # 0 = disabled (default). When > 0, train.py builds a gradient mask so only
    # params with 'lora_A'/'lora_B' in their path receive nonzero optimizer
    # updates -- the base backbone weights are mathematically frozen even
    # though JAX still computes (and discards) a gradient for them.
    lora_rank: int = 0
    lora_alpha: float = None  # defaults to lora_rank (scaling factor = 1.0) if unset

    # Decoder objective
    decoder_prob: float = 0.5  # Probability of decoder (CE) step vs denoiser (L2) step
    decoder_noise_scale: float = 1.0  # Scale of noise in logit-normal-noised latent for CE branch
    decoder_p_mean: float = 0.8  # Mean for logit-normal noise schedule in decoder objective
    decoder_p_std: float = 0.8  # Std for logit-normal noise schedule in decoder objective

    # Conditioning / CFG
    label_drop_prob: float = 0.0
    self_cond_prob: float = 0.5
    self_cond_cfg_min: float = 0.5
    self_cond_cfg_max: float = 5.0

    # Training (optimizer + schedule)
    epochs: int = 200
    warmup_epochs: float = None
    warmup_steps: int = 5000
    batch_size: int = None
    global_batch_size: int = 512
    lr: float = None
    blr: float = 5e-5
    min_lr: float = 0.0
    lr_schedule: str = "constant"
    weight_decay: float = 0.0
    optimizer: str = "muon"  # "adamw" or "muon"
    adam_b1: float = 0.9
    adam_b2: float = 0.95
    grad_accum_steps: int = 1  # Gradient accumulation steps (optimizer updates every K mini-batches)
    generation_batch_size: int = None  # Override batch size for generation eval (defaults to global_batch_size)
    max_steps: int = None              # Hard cap on total training steps (overrides epochs)

    # EMA
    ema_decay1: float = 0.9999

    # Sampling
    sampling_configs_path: str = None
    # Sampling configs sweep (list of SamplingConfig objects, loaded from YAML)
    sampling_configs: list = [SamplingConfig()]
    num_samples: int = 100
    # Decode-time noise (optional): perturb the final latent before decoding to
    # slightly increase output entropy/diversity. Off by default — no effect on
    # existing behavior/results unless explicitly enabled.
    addnoise: bool = False
    addnoise_scale: float = 0.02  # Std of the Gaussian noise added when addnoise=True

    # PPL Evaluation
    online_eval: bool = True  # Enable PPL evaluation for generated samples
    eval_ppl_model: str = "gpt2-large"  # Model for PPL evaluation
    eval_ppl_batch_size: int = 64  # Batch size for PPL evaluation (adjusted to be divisible by device count)
    eval_ppl_max_length: int = 1024  # Max sequence length for PPL evaluation

    # MAUVE / REP-n (unconditional generation only; opt-in, off by default)
    eval_mauve: bool = False  # Also compute MAUVE (vs. a real-text reference sample) and REP-4
    eval_mauve_featurize_model: str = "gpt2-large"  # Featurizer for MAUVE's GPT-2 embeddings
    eval_mauve_device_id: int = 0  # torch device index for MAUVE featurization (-1 = CPU)
    # GPT-2 token truncation window for MAUVE featurization. Dataset-specific:
    # OWT's corrected protocol uses 768 (symmetric truncation of ~1024-token
    # documents avoids the last-token-mismatch collapse seen at the old
    # default of 1024 -- see docs/lm1b_mauve_debug.md and the OWT MAUVE fix).
    # LM1B's own sentence-level examples average ~28 GPT-2 tokens -- far
    # below any truncation window -- so 96 (matching the external LM1B
    # MAUVE convention) is a reasonable default there, though per
    # docs/lm1b_mauve_debug.md this window is NOT the dominant factor in
    # LM1B's low score. Default (1024) preserves prior behavior for any
    # config that doesn't set this explicitly.
    eval_mauve_max_length: int = 1024

    # First-sentence truncation for unconditional generation (opt-in, off by default).
    # LM1B's preprocessing never appends an EOS token (add_special_tokens=False), so
    # the decoder CE loss never trains the model to terminate: generation runs to
    # max_length, producing several real-looking sentences concatenated into one
    # run-on blob instead of LM1B's single-sentence examples. Random length-matched
    # truncation does NOT fix this (verified empirically: MAUVE stayed at ~0.007,
    # same as the untruncated run, because a random cut still lands mid-clause).
    # Cutting at the first sentence boundary does fix it (MAUVE 0.0097 -> 0.095 in
    # the same test) since the model's first embedded sentence is usually well-formed
    # on its own. Requires nltk's punkt tokenizer.
    gen_truncate_to_first_sentence: bool = False

    # Logging & Checkpointing
    log_freq: int = 100
    eval_freq: int = 10
    save_freq: float = 100  # Can be fractional (e.g., 0.1 for saving every 0.1 epoch)

    # Output
    output_dir: str = "./output_dir"
    hf_repo_id: str = None  # Optional HF repo id to mirror local outputs/checkpoints.
    resume: str = None

    # Wandb
    use_wandb: bool = False
    wandb_project: str = "ELF"
    wandb_entity: str = None
    wandb_run_name: str = None
    wandb_tag: str = None
    wandb_resume: str = "allow"

    # Misc
    seed: int = 0
    num_workers: int = 0


def load_config_from_yaml(path: str) -> Config:
    """Load a YAML config and override defaults in Config."""
    config = Config()
    if not path or not os.path.isfile(path):
        return config

    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f) or {}

    for key, value in cfg_dict.items():
        if key == "sampling_configs":
            continue  # handled below
        if hasattr(config, key):
            setattr(config, key, value)

    if config.sampling_configs_path:
        config.sampling_configs = load_sampling_configs(config.sampling_configs_path)

    return config


def apply_config_overrides(config: Config, overrides: list) -> Config:
    """Apply command-line config overrides to a Config object.
    
    Args:
        config: Config object to modify
        overrides: List of strings in format "field_name=value"
    
    Returns:
        Modified config object
    """
    if not overrides:
        return config
    
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override format: '{override}'. Expected 'field_name=value'")
        
        field_name, value_str = override.split("=", 1)
        field_name = field_name.strip()
        value_str = value_str.strip()
        
        if not hasattr(config, field_name):
            raise ValueError(f"Config has no field named '{field_name}'")
        
        original_value = getattr(config, field_name)
        original_type = type(original_value)

        # Allow setting a field back to None
        if value_str.lower() == "none":
            setattr(config, field_name, None)
            continue

        if original_value is None:
            # Use type annotation to infer the intended type
            annotated_type = config.__annotations__.get(field_name)
            if annotated_type == int:
                converted_value = int(value_str)
            elif annotated_type == float:
                converted_value = float(value_str)
            elif annotated_type == bool:
                converted_value = value_str.lower() in ("true", "1", "yes")
            else:
                converted_value = value_str
        elif original_type == bool:
            converted_value = value_str.lower() in ("true", "1", "yes")
        elif original_type == int:
            converted_value = int(value_str)
        elif original_type == float:
            converted_value = float(value_str)
        elif original_type == str:
            converted_value = value_str
        else:
            converted_value = value_str
        
        setattr(config, field_name, converted_value)

    return config


def load_sampling_configs(sampling_configs_path: str):
    """Return sampling configs, loading from sampling_configs_path if set."""
    with open(sampling_configs_path, "r") as f:
        entries = yaml.safe_load(f)
    return [SamplingConfig(**entry) for entry in entries]
