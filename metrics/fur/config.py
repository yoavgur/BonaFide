"""FUR hyperparameters and configuration.

Default values from Table 7 of the paper and reference const.py.

Origin: Adapted from https://github.com/technion-cs-nlp/parametric-faithfulness/blob/main/const.py
Changes: Restructured as a dataclass; added model_lr_overrides from dataset_model_best_lr.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FURConfig:
    """Configuration for FUR metric."""

    # NPO+KL loss hyperparameters (Table 7)
    beta: float = 0.1
    npo_coeff: float = 1.0
    kl_coeff: float = 1.0

    # Training
    num_epochs: int = 5  # Matching v2 default (was 15)
    warmup: bool = False  # Paper deviates from NPO default
    learning_rate: float = 1e-5  # Default; override per model via model_lr_overrides
    batch_size: int = 1

    # FF2-only parameter selection
    # Pattern uses substring matching via `in`, so "down_proj" matches:
    #   - Dense models:           mlp.down_proj.weight
    #   - Sparse-MoE models:      mlp.experts.N.down_proj.weight
    #   - Fused-MoE (Qwen3-30B):  mlp.experts.down_proj  (no .weight suffix; fused tensor)
    ff2_only: bool = True
    ff2_param_pattern: str = "down_proj"

    # Segmentation
    min_content_tokens: int = 2  # Steps with fewer content words are skipped

    # Retain set (used during NPO+KL training; D_RT in the paper)
    retain_sample_count: int = 4  # Number of other CoT instances for retain set

    # Specificity set (paper Eq. 3 D_s; held-out, NEVER seen by the optimizer).
    # We compute binary argmax-agreement on the next-token-after-prefill for each
    # instance, before vs after unlearning. Disjoint from retain by construction.
    # Matches retain_sample_count by default — same cost as before, just held-out.
    specificity_set_size: int = 4
    specificity_max_input_len: int = 4096  # Truncate spec prefixes to this length
    # Prefill string appended after `prompt + cot` so the argmax we measure lands
    # on the first *content* token of the answer (not the structural `{` / `</think>`).
    # Matches the project's JSON answer format from generation/model_registry.py.
    specificity_answer_prefill: str = '{"final_answer": "'

    # MMLU sanity check (paper §6.1 "general capabilities"). Tests whether
    # unlearning damages the model's general knowledge by comparing argmax-letter
    # predictions on a small bundled MMLU sample, before vs after each step's
    # unlearning. Reported as `mmlu_agreement` in the per-step metadata.
    compute_mmlu_check: bool = False
    mmlu_set_size: int = 10  # Number of MMLU questions (capped at bundled set size of 10)

    # Gradient clipping
    max_grad_norm: float = 1.0  # Clip gradient norm to prevent catastrophic unlearning

    # Paraphrased forget/retain sets (v2 feature)
    num_paraphrases: int = 0  # 0 = disabled. Semantic paraphrases per step added to forget/retain sets.
    paraphrase_model: str = "gemini-3.1-flash-lite-preview"

    # Memory management
    max_seq_len: int = 1024  # Truncate forget/retain sequences to this length
    kl_chunk_size: int = 32  # Chunk size for KL divergence computation
    keep_oracle_on_gpu: bool = True  # Keep oracle FF2 weights on GPU (faster, more VRAM)
    greedy_generation: bool = False  # Force greedy decoding for post-unlearning generation
    # WARNING: greedy causes degenerate loops on thinking models (Qwen3-Thinking, etc.)
    num_generation_samples: int = 5  # Generate N samples per step, majority vote on answer_changed

    # vLLM-accelerated generation (two-stage): unlearn all steps with HF first,
    # then destroy HF model and generate all responses with vLLM.
    # Only applies to STEP scoring; COT scoring uses HF generation (needs early stopping).
    use_vllm_generation: bool = False  # Set True by CLI/orchestrator
    use_shm_pipeline: bool = False    # Pipeline state dict saves via /dev/shm (set by CLI/orchestrator)
    fur_batch_size: int = 10          # Default/max steps per batch in two-stage pipeline
    fur_auto_batch: bool = True       # Auto-size batch from available disk space (capped by fur_batch_size)
    gpu_memory_utilization: float = 0.9  # vLLM gpu_memory_utilization for post-unlearning generation

    # Validation metrics (paper Eq. 2 efficacy + Eq. 3 specificity proxy)
    compute_validation: bool = False  # Compute per-step efficacy / specificity after unlearning

    # Skip post-unlearning generation entirely. Useful for cheaply verifying
    # whether unlearning is actually working (efficacy/specificity only), without
    # paying for the generation step. When True, answer_changed in results is unset.
    # Implies compute_validation=True and forces use_vllm_generation=False.
    validation_only: bool = False

    # Per-model learning rate overrides (from reference const.py dataset_model_best_lr).
    # Keys are model short names, values are learning rates.
    model_lr_overrides: dict = field(default_factory=lambda: {
        # From reference const.py dataset_model_best_lr
        "Phi-3": 1e-04,
        "LLaMA-3": 1e-05,
        "LLaMA-3-3B": 3e-05,
        "Mistral-2": 5e-06,
        # Not in paper — tuned to avoid loss explosion
        "Qwen3-4B-Thinking": 2.25e-5,
        "Qwen3-4B-Instruct": 1e-5,
    })
