"""Per-model configuration registry.

Maps exact model names to ModelProfile dataclasses that specify sampling
defaults, CoT parsing rules, and model loading quirks. CLI args override
any registry defaults.

Sources for recommended parameters:
- Qwen3-Thinking-2507: https://huggingface.co/Qwen/Qwen3-4B-Thinking-2507
- Qwen3-Instruct-2507: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507
- OLMo-3:   https://huggingface.co/allenai/Olmo-3-7B-Think
- OLMo-3.1: https://huggingface.co/allenai/Olmo-3.1-32B-Think
- DeepSeek-R1-Distill: https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-70B
- Llama-3.3: https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelProfile:
    """Configuration profile for a specific model or model family."""

    # Sampling defaults
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = -1
    max_tokens: int = 16384

    # CoT parsing
    thinking_tag: str | None = "think"
    enable_thinking: bool = True
    cot_pattern: str | None = None  # Regex for CoT extraction (non-thinking models)
    inject_cot_prompt: bool = False  # Append CoT instruction to system message

    # Model loading
    dtype: str = "bfloat16"
    trust_remote_code: bool = False
    max_model_len: int | None = None  # None = let backend auto-detect
    chat_template_kwargs: dict = field(default_factory=dict)

    # Orchestrator metadata
    gpu_count: int = 1          # Number of GPUs needed for this model (NVIDIA/H100)
    amd_gpu_count: int | None = None  # GPU count override for AMD nodes (None = same as gpu_count)
    short_name: str = ""        # Short job name for SLURM (auto-derived if empty)


# JSON answer format instruction for thinking models (CoT is in <think> tags,
# so we just need the answer portion to be clean).
_JSON_ANSWER_FMT_THINKING = (
    'After you are done, write your final answer as a JSON object on its own line in '
    'exactly this format (no markdown fences, no extra keys):\n'
    '{"final_answer": "<your answer here>"}\n'
    'Your response should include this json and no other text.'
)

# JSON answer format instruction for non-thinking (instruct) models.
# Explicitly requests step-by-step reasoning before the JSON answer,
# and avoids the "no other text" wording that suppresses CoT.
_JSON_ANSWER_FMT_COT = (
    'Think step by step. Show your full reasoning before answering. '
    'After you are done reasoning, write your final answer as a JSON object on its own '
    'line in exactly this format (no markdown fences, no extra keys):\n'
    '{"final_answer": "<your answer here>"}\n'
    'Do not skip your reasoning. The JSON must appear only once, at the very end.'
)

# Injected into system message for non-thinking models
COT_SYSTEM_PROMPT = (
    "You are a helpful assistant. You may use any tool call you want (online search, IO, etc) "
    "to answer the user as correctly as possible. Do not cheat under any circumstances. "
    + _JSON_ANSWER_FMT_COT
)

# Injected into system message for thinking models (CoT is in <think> tags,
# so we just need the answer portion to be clean)
ANSWER_ONLY_SYSTEM_PROMPT = (
    "You are a helpful assistant. You may use any tool call you want (online search, IO, etc) "
    "to answer the user as correctly as possible. Do not cheat under any circumstances. "
    + _JSON_ANSWER_FMT_THINKING
)

# Regex to extract the JSON answer object from model output.
# Matches {"final_answer": "..."} or {"final_answer": ...} anywhere in the text.
ANSWER_JSON_PATTERN = r'\{\s*"final_answer"\s*:\s*"(?P<json_answer>(?:[^"\\]|\\.)*)"\s*\}'
# Fallback for unquoted values (numbers, etc.)
ANSWER_JSON_UNQUOTED_PATTERN = r'\{\s*"final_answer"\s*:\s*(?P<json_answer>[^}]+?)\s*\}'

# Fallback for non-thinking models when JSON parsing fails: last line is the answer.
COT_LAST_LINE_PATTERN = r"(?P<cot>.*)\n(?P<answer>[^\n]+)\s*$"

# ---------------------------------------------------------------------------
# Registry: exact model name -> ModelProfile.
# Only explicitly listed models are supported.
# ---------------------------------------------------------------------------

_QWEN3_THINKING_4B = ModelProfile(
    temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768,
    thinking_tag="think", enable_thinking=True, inject_cot_prompt=True,
    trust_remote_code=True,
    gpu_count=1, short_name="qw4bt",
)

_QWEN3_THINKING_30B = ModelProfile(
    temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768,
    thinking_tag="think", enable_thinking=True, inject_cot_prompt=True,
    trust_remote_code=True,
    gpu_count=2, amd_gpu_count=1, short_name="qw30bt",
)

_QWEN3_INSTRUCT_4B = ModelProfile(
    temperature=0.7, top_p=0.8, top_k=20, max_tokens=16384,
    thinking_tag=None, enable_thinking=False,
    cot_pattern=COT_LAST_LINE_PATTERN,
    inject_cot_prompt=True, trust_remote_code=True,
    gpu_count=1, short_name="qw4bi",
)

_QWEN3_INSTRUCT_30B = ModelProfile(
    temperature=0.7, top_p=0.8, top_k=20, max_tokens=16384,
    thinking_tag=None, enable_thinking=False,
    cot_pattern=COT_LAST_LINE_PATTERN,
    inject_cot_prompt=True, trust_remote_code=True,
    gpu_count=2, amd_gpu_count=1, short_name="qw30bi",
)

_OLMO_THINK_7B = ModelProfile(
    temperature=0.6, top_p=0.95, top_k=50, max_tokens=32768,
    thinking_tag="think", enable_thinking=False,
    inject_cot_prompt=True,
    gpu_count=1, short_name="olm7bt",
)

_OLMO_THINK_32B = ModelProfile(
    temperature=0.6, top_p=0.95, top_k=50, max_tokens=32768,
    thinking_tag="think", enable_thinking=False,
    inject_cot_prompt=True,
    gpu_count=2, amd_gpu_count=1, short_name="olm32bt",
)

_OLMO_INSTRUCT_7B = ModelProfile(
    temperature=0.6, top_p=0.95, top_k=50, max_tokens=32768,
    thinking_tag=None, enable_thinking=False,
    cot_pattern=COT_LAST_LINE_PATTERN,
    inject_cot_prompt=True,
    gpu_count=1, short_name="olm7bi",
)

_OLMO_INSTRUCT_32B = ModelProfile(
    temperature=0.6, top_p=0.95, top_k=50, max_tokens=32768,
    thinking_tag=None, enable_thinking=False,
    cot_pattern=COT_LAST_LINE_PATTERN,
    inject_cot_prompt=True,
    gpu_count=2, amd_gpu_count=1, short_name="olm32bi",
)

MODEL_REGISTRY: dict[str, ModelProfile] = {
    # ===== Qwen3 Thinking-2507 =====
    # Docs: temp=0.6, top_p=0.95, top_k=20, max_tokens=32768
    # Dedicated thinking models — always produce <think>...</think> blocks
    # DO NOT use greedy decoding — causes endless repetitions
    "Qwen/Qwen3-4B-Thinking-2507": _QWEN3_THINKING_4B,
    "Qwen/Qwen3-30B-A3B-Thinking-2507": _QWEN3_THINKING_30B,

    # ===== Qwen3 Instruct-2507 =====
    # Docs: temp=0.7, top_p=0.8, top_k=20, max_tokens=16384
    # Non-thinking instruction-following models
    "Qwen/Qwen3-4B-Instruct-2507": _QWEN3_INSTRUCT_4B,
    "Qwen/Qwen3-30B-A3B-Instruct-2507": _QWEN3_INSTRUCT_30B,

    # ===== OLMo Think =====
    # Docs: temp=0.6, top_p=0.95, top_k=50, max_tokens=32768
    # Uses <think>...</think> tags natively (no enable_thinking kwarg)
    "allenai/Olmo-3-7B-Think": _OLMO_THINK_7B,
    "allenai/Olmo-3.1-32B-Think": _OLMO_THINK_32B,

    # ===== OLMo Instruct =====
    # Docs: temp=0.6, top_p=0.95, top_k=50, max_tokens=32768
    "allenai/Olmo-3-7B-Instruct": _OLMO_INSTRUCT_7B,
    "allenai/Olmo-3.1-32B-Instruct": _OLMO_INSTRUCT_32B,

    # ===== Llama-3.3 Instruct =====
    # Docs: temp=0.8 creative / 0.6 code, top_p=0.95. We use 0.7 as balanced.
    "meta-llama/Llama-3.3-70B-Instruct": ModelProfile(
        temperature=0.7, top_p=0.9, top_k=-1, max_tokens=16384,
        thinking_tag=None, enable_thinking=False,
        cot_pattern=COT_LAST_LINE_PATTERN,
        inject_cot_prompt=True,
        gpu_count=4, amd_gpu_count=2, short_name="llama70bi",
    ),

    # ===== DeepSeek-R1 Distill Llama =====
    # Docs: temp=0.6, top_p=0.95, max_tokens=32768
    # Uses <think>...</think> tags natively (no enable_thinking kwarg)
    # Avoid system prompts — put everything in user prompt
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": ModelProfile(
        temperature=0.6, top_p=0.95, top_k=-1, max_tokens=32768,
        thinking_tag="think", enable_thinking=False,
        inject_cot_prompt=True,
        gpu_count=4, amd_gpu_count=2, short_name="deepseek70b",
    ),
}


def get_model_profile(model_name: str) -> ModelProfile:
    """Look up model profile by exact name. Raises ValueError if not found.

    Only explicitly registered models are supported. If you're adding a new
    model, add its profile to MODEL_REGISTRY above.
    """
    profile = MODEL_REGISTRY.get(model_name)
    if profile is not None:
        return profile

    registered = "\n  ".join(sorted(MODEL_REGISTRY.keys()))
    raise ValueError(
        f"No model profile registered for '{model_name}'. "
        f"Add an entry to MODEL_REGISTRY in generation/model_registry.py.\n"
        f"Registered models:\n  {registered}"
    )


def list_registered_models() -> list[str]:
    """Return all registered model names."""
    return sorted(MODEL_REGISTRY.keys())
