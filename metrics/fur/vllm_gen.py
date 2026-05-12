"""vLLM-accelerated generation for FUR two-stage pipeline.

In the two-stage approach:
  Stage 1 (parent process): HF model unlearns all steps, saves modified FF2 weights.
  Stage 2 (subprocess):     Fresh process loads vLLM, patches weights, generates.

The subprocess approach is necessary because the parent's CUDA context holds
residual memory that cannot be freed (driver-level limitation). A fresh process
gets a clean CUDA context with full GPU memory available.

Usage as subprocess:
    python -m metrics.fur.vllm_gen --manifest /path/to/manifest.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections import OrderedDict

import torch

logger = logging.getLogger(__name__)


class VLLMGenerator:
    """vLLM instance for fast generation with weight patching.

    Loads the base model into vLLM. After loading, call patch_weights() with
    modified FF2 state dicts, then generate().
    """

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: int = 32768,
    ) -> None:
        logger.info(
            "Initializing vLLM (model=%s, tp=%d, mem_util=%.2f, max_model_len=%d)",
            model_name, tensor_parallel_size, gpu_memory_utilization, max_model_len,
        )

        # Disable V1 multiprocessing so model is in-process for weight patching.
        # Only works for TP=1; for TP>1, vLLM uses multiproc anyway and we use
        # collective_rpc instead.
        if tensor_parallel_size == 1:
            os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
        else:
            # collective_rpc needs to serialize our weight-patching function.
            # V1's default msgpack serializer can't handle functions — enable
            # pickle fallback. Safe here since we control the subprocess.
            os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        from vllm import LLM

        self.llm = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            dtype="bfloat16",
            trust_remote_code=True,
        )
        self._tp_size = tensor_parallel_size

        # For TP=1 (in-process): direct model access for weight patching.
        # For TP>1 (multiproc): use collective_rpc to patch in worker processes.
        if tensor_parallel_size == 1:
            self._vllm_model = (
                self.llm.llm_engine.engine_core  # InprocClient
                .engine_core                      # EngineCore
                .model_executor                   # UniProcExecutor
                .driver_worker                    # WorkerWrapperBase → worker
                .model_runner.model
            )
        else:
            self._vllm_model = None  # Use collective_rpc instead

        logger.info("vLLM model ready (tp=%d)", tensor_parallel_size)

    def patch_weights_from_file(self, state_dict_path: str) -> None:
        """Patch modified FF2 (down_proj) weights from a file on disk.

        For TP=1: loads once in-process and applies directly.
        For TP>1: each worker loads from the same file independently via
        collective_rpc (no redundant temp file copies).
        """
        if self._vllm_model is not None:
            # TP=1: load once, apply in-process
            sd = torch.load(state_dict_path, weights_only=True)
            self._vllm_model.load_weights(weights=list(sd.items()))
        else:
            # TP>1: each worker loads directly from the existing file
            def _load_weights_from_file(worker, path):
                import torch as _torch
                sd = _torch.load(path, weights_only=True)
                worker.model_runner.model.load_weights(weights=list(sd.items()))

            self.llm.collective_rpc(_load_weights_from_file, args=(state_dict_path,))

    def generate(
        self,
        prompt_str: str,
        max_new_tokens: int,
        temperature: float = 0.0,
        top_p: float = 1.0,
        top_k: int = -1,
        n: int = 1,
    ) -> list[str]:
        """Generate n responses using vLLM (batched, single prompt).

        Args:
            n: Number of completions to generate. With n>1 and temperature>0,
               each completion samples independently (different outputs).

        Returns:
            List of n raw output strings.
        """
        from vllm import SamplingParams

        params = SamplingParams(
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k if top_k > 0 else -1,
            n=n,
        )

        outputs = self.llm.generate([prompt_str], sampling_params=[params])

        if not outputs or not outputs[0].outputs:
            raise RuntimeError("vLLM returned no outputs for generation")

        results = []
        for completion in outputs[0].outputs:
            logger.debug(
                "vLLM generated %d tokens, finish_reason=%s",
                len(completion.token_ids), completion.finish_reason,
            )
            results.append(completion.text)
        return results


# ---------------------------------------------------------------------------
# Subprocess entry point
# ---------------------------------------------------------------------------

def run_generation_subprocess(manifest_path: str) -> None:
    """Run vLLM generation from a manifest file. Called as a subprocess.

    The manifest JSON contains model config and a list of steps, each with
    a state_dict_path and generation parameters. Results are written as
    individual JSON files next to each state dict (crash-safe incremental).
    """
    with open(manifest_path) as f:
        manifest = json.load(f)

    gen = VLLMGenerator(
        model_name=manifest["model_name"],
        tensor_parallel_size=manifest["tensor_parallel_size"],
        gpu_memory_utilization=manifest.get("gpu_memory_utilization", 0.9),
        max_model_len=manifest.get("max_model_len", 32768),
    )

    steps = manifest["steps"]
    logger.info("Generating %d steps with vLLM", len(steps))

    for i, entry in enumerate(steps):
        result_path = entry["state_dict_path"].replace(".pt", "_result.json")

        # Resume: skip steps that already have results
        if os.path.exists(result_path):
            logger.info("Step %d/%d: result exists, skipping", i + 1, len(steps))
            continue

        if entry.get("skipped", False):
            # Step was skipped during unlearning (too few content tokens)
            result = {"skipped": True}
            with open(result_path, "w") as f:
                json.dump(result, f)
            logger.info("Step %d/%d: skipped (too few tokens)", i + 1, len(steps))
            continue

        t0 = time.time()

        # Patch weights directly from file (no redundant copy)
        gen.patch_weights_from_file(entry["state_dict_path"])
        t_patch = time.time()

        # Generate (n samples for majority vote)
        n_samples = entry.get("num_samples", 1)
        raw_outputs = gen.generate(
            prompt_str=entry["prompt_str"],
            max_new_tokens=entry["max_new_tokens"],
            temperature=entry.get("temperature", 0.0),
            top_p=entry.get("top_p", 1.0),
            top_k=entry.get("top_k", -1),
            n=n_samples,
        )
        t_gen = time.time()

        logger.info(
            "Step %d/%d: patch=%.1fs, gen=%.1fs (n=%d), total=%.1fs",
            i + 1, len(steps), t_patch - t0, t_gen - t_patch, n_samples, t_gen - t0,
        )

        # Write result atomically (write to tmp then rename)
        result = {"raw_outputs": raw_outputs, "wall_time_s": round(t_gen - t0, 3)}
        tmp_path = result_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(result, f)
        os.replace(tmp_path, result_path)

        # Delete state dict — result is saved, no longer needed
        try:
            os.remove(entry["state_dict_path"])
        except OSError:
            pass

    logger.info("All %d steps completed", len(steps))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="FUR vLLM generation subprocess")
    parser.add_argument("--manifest", required=True, help="Path to generation manifest JSON")
    args = parser.parse_args()

    run_generation_subprocess(args.manifest)
