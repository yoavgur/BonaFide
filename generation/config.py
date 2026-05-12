"""GenerationConfig: stores everything needed to reproduce a generation run."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class GenerationConfig:
    """Full reproducibility manifest for a generation run.

    Saved as run_config.json alongside results.
    """

    # --- Identity ---
    run_id: str = ""
    timestamp: str = ""

    # --- Model ---
    model: str = ""
    model_revision: str = ""  # Git SHA from HF Hub
    dtype: str = "bfloat16"
    trust_remote_code: bool = False

    # --- Backend ---
    backend: str = "vllm"  # "vllm" or "hf"
    backend_version: str = ""
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None

    # --- Execution ---
    batch_size: int = 50  # Process in chunks with incremental saves (0 = all at once)
    resume: bool = False

    # --- Sampling ---
    seed: int = 42
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = -1
    max_tokens: int = 16384

    # --- Prompt ---
    thinking_tag: str | None = "think"
    enable_thinking: bool = True
    cot_pattern: str | None = None
    inject_cot_prompt: bool = True  # Append CoT instruction to system message
    chat_template_kwargs: dict = field(default_factory=dict)

    # --- No-CoT mode ---
    # When True, suppress chain-of-thought generation by closing the thinking
    # block with a single filler token (.) and prefilling the start of the
    # JSON answer. Per-row answer key comes from the input CSV's `answer_key`
    # column ("final_answer" by default, "final_node" for graph rows).
    no_cot: bool = False

    # --- Data columns ---
    prompt_column: str = "prompt"
    system_message_column: str = "system_message"
    id_column: str = "id"
    dataset_column: str = "dataset"

    # --- Input ---
    input_path: str = ""
    input_sha256: str = ""
    num_rows: int = 0

    # --- Output ---
    output_path: str = ""  # Full path to output CSV (auto-derived if not set)

    # --- Environment ---
    torch_version: str = ""
    cuda_version: str | None = None
    python_version: str = ""
    gpu_names: list[str] = field(default_factory=list)
    num_gpus: int = 0

    def populate_environment(self) -> None:
        """Auto-detect and fill environment fields."""
        import torch

        self.python_version = platform.python_version()
        self.torch_version = torch.__version__
        self.cuda_version = torch.version.cuda
        if torch.cuda.is_available():
            self.num_gpus = torch.cuda.device_count()
            self.gpu_names = [
                torch.cuda.get_device_name(i) for i in range(self.num_gpus)
            ]

    def populate_backend_version(self) -> None:
        """Detect and fill backend library version."""
        if self.backend == "vllm":
            import vllm

            self.backend_version = f"vllm=={vllm.__version__}"
        else:
            import transformers

            self.backend_version = f"transformers=={transformers.__version__}"

    def populate_model_revision(self) -> None:
        """Fetch the model's git SHA from HuggingFace Hub."""
        from huggingface_hub import model_info

        info = model_info(self.model)
        self.model_revision = info.sha or ""

    def populate_input_hash(self) -> None:
        """Compute SHA256 of the input file."""
        path = Path(self.input_path)
        if path.exists():
            sha = hashlib.sha256()
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha.update(chunk)
            self.input_sha256 = sha.hexdigest()

    def generate_run_id(self) -> None:
        """Generate a run ID from model name, seed, and timestamp."""
        model_short = self.model.split("/")[-1].lower()
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.timestamp = datetime.now(timezone.utc).isoformat()
        self.run_id = f"{model_short}_{self.seed}_{ts}"

    def save(self, path: Path | None = None) -> Path:
        """Save config as JSON alongside the output CSV."""
        if path is None:
            csv_path = Path(self.output_path)
            path = csv_path.with_name(csv_path.stem + "_run_config.json")
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)
        return path

    @classmethod
    def load(cls, path: Path) -> GenerationConfig:
        """Load config from JSON."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)
