"""Tests for the generation pipeline.

Run with: pytest generation/tests.py -v
"""

from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from generation.backends import (
    GenerationResult,
    _row_seed,
    _set_all_seeds,
    build_prompt,
)
from generation.config import GenerationConfig
from generation.generate import (
    _apply_profile_defaults,
    _derive_output_path,
    _load_completed_ids,
    _load_input,
    _save_results,
)
from generation.model_registry import (
    MODEL_REGISTRY,
    ModelProfile,
    get_model_profile,
    list_registered_models,
)
from generation.thinking import split_thinking


# =========================================================================
# Model Registry
# =========================================================================


class TestModelRegistry:
    """Tests for model_registry.py"""

    def test_exactly_10_models_registered(self):
        assert len(MODEL_REGISTRY) == 10

    def test_all_expected_models_present(self):
        expected = [
            "Qwen/Qwen3-4B-Thinking-2507",
            "Qwen/Qwen3-4B-Instruct-2507",
            "Qwen/Qwen3-30B-A3B-Thinking-2507",
            "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "allenai/Olmo-3-7B-Think",
            "allenai/Olmo-3-7B-Instruct",
            "allenai/Olmo-3.1-32B-Think",
            "allenai/Olmo-3.1-32B-Instruct",
            "meta-llama/Llama-3.3-70B-Instruct",
            "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        ]
        for model in expected:
            assert model in MODEL_REGISTRY, f"Missing: {model}"

    def test_unregistered_model_raises(self):
        with pytest.raises(ValueError, match="No model profile registered"):
            get_model_profile("some-random/unknown-model")

    def test_unregistered_model_error_lists_models(self):
        with pytest.raises(ValueError, match="Registered models"):
            get_model_profile("totally/fake")

    def test_old_qwen3_4b_not_supported(self):
        """Old Qwen/Qwen3-4B (pre-2507) should not match."""
        with pytest.raises(ValueError):
            get_model_profile("Qwen/Qwen3-4B")

    # --- Qwen3 Thinking ---

    def test_qwen3_4b_thinking(self):
        p = get_model_profile("Qwen/Qwen3-4B-Thinking-2507")
        assert p.thinking_tag == "think"
        assert p.enable_thinking is True
        assert p.temperature == 0.6
        assert p.top_p == 0.95
        assert p.top_k == 20
        assert p.max_tokens == 32768
        assert p.trust_remote_code is True
        assert p.inject_cot_prompt is True
        assert p.cot_pattern is None

    def test_qwen3_30b_a3b_thinking(self):
        p = get_model_profile("Qwen/Qwen3-30B-A3B-Thinking-2507")
        assert p.thinking_tag == "think"
        assert p.enable_thinking is True
        assert p.top_k == 20
        assert p.trust_remote_code is True

    def test_qwen3_thinking_models_share_profile(self):
        p1 = get_model_profile("Qwen/Qwen3-4B-Thinking-2507")
        p2 = get_model_profile("Qwen/Qwen3-30B-A3B-Thinking-2507")
        assert p1 is p2

    # --- Qwen3 Instruct ---

    def test_qwen3_4b_instruct(self):
        from generation.model_registry import COT_LAST_LINE_PATTERN
        p = get_model_profile("Qwen/Qwen3-4B-Instruct-2507")
        assert p.thinking_tag is None
        assert p.enable_thinking is False
        assert p.temperature == 0.7
        assert p.top_p == 0.8
        assert p.top_k == 20
        assert p.max_tokens == 16384
        assert p.trust_remote_code is True
        assert p.inject_cot_prompt is True
        assert p.cot_pattern == COT_LAST_LINE_PATTERN

    def test_qwen3_30b_a3b_instruct(self):
        p = get_model_profile("Qwen/Qwen3-30B-A3B-Instruct-2507")
        assert p.thinking_tag is None
        assert p.enable_thinking is False

    def test_qwen3_instruct_models_share_profile(self):
        p1 = get_model_profile("Qwen/Qwen3-4B-Instruct-2507")
        p2 = get_model_profile("Qwen/Qwen3-30B-A3B-Instruct-2507")
        assert p1 is p2

    # --- OLMo ---

    def test_olmo3_7b_think(self):
        p = get_model_profile("allenai/Olmo-3-7B-Think")
        assert p.thinking_tag == "think"
        assert p.enable_thinking is False  # OLMo doesn't use this kwarg
        assert p.temperature == 0.6
        assert p.top_p == 0.95
        assert p.top_k == 50
        assert p.max_tokens == 32768

    def test_olmo3_7b_instruct(self):
        p = get_model_profile("allenai/Olmo-3-7B-Instruct")
        assert p.thinking_tag is None
        assert p.enable_thinking is False
        assert p.temperature == 0.6
        assert p.top_k == 50

    def test_olmo31_32b_think(self):
        p = get_model_profile("allenai/Olmo-3.1-32B-Think")
        assert p.thinking_tag == "think"
        assert p.enable_thinking is False
        assert p.top_k == 50

    def test_olmo31_32b_instruct(self):
        p = get_model_profile("allenai/Olmo-3.1-32B-Instruct")
        assert p.thinking_tag is None
        assert p.enable_thinking is False

    # --- DeepSeek ---

    def test_deepseek_r1_distill_llama(self):
        p = get_model_profile("deepseek-ai/DeepSeek-R1-Distill-Llama-70B")
        assert p.thinking_tag == "think"
        assert p.enable_thinking is False  # DeepSeek doesn't use this kwarg
        assert p.temperature == 0.6
        assert p.top_p == 0.95
        assert p.top_k == -1
        assert p.max_tokens == 32768

    # --- Llama ---

    def test_llama_33_70b_instruct(self):
        p = get_model_profile("meta-llama/Llama-3.3-70B-Instruct")
        assert p.thinking_tag is None
        assert p.enable_thinking is False
        assert p.temperature == 0.7
        assert p.top_p == 0.9
        assert p.top_k == -1
        assert p.max_tokens == 16384

    # --- Utility ---

    def test_list_registered_models(self):
        models = list_registered_models()
        assert len(models) == 10
        assert all(isinstance(m, str) for m in models)
        assert models == sorted(models)  # Should be sorted


# =========================================================================
# Thinking / CoT extraction
# =========================================================================


class TestSplitThinking:
    """Tests for thinking.py"""

    def test_empty_input(self):
        assert split_thinking("", "think") == ("", "")
        assert split_thinking("", None) == ("", "")

    def test_no_thinking_tag(self):
        cot, answer = split_thinking("The answer is 42.", None)
        assert cot == ""
        assert answer == "The answer is 42."

    def test_normal_think_tags(self):
        raw = "<think>Step 1: reason.\nStep 2: conclude.</think>The answer is 4."
        cot, answer = split_thinking(raw, "think")
        assert cot == "Step 1: reason.\nStep 2: conclude."
        assert answer == "The answer is 4."

    def test_truncated_no_closing_tag(self):
        raw = "<think>I'm thinking really hard about this and"
        cot, answer = split_thinking(raw, "think")
        assert "thinking really hard" in cot
        assert answer == ""

    def test_truncated_no_tags_at_all(self):
        raw = "Just some raw text with no tags"
        cot, answer = split_thinking(raw, "think")
        assert cot == raw
        assert answer == ""

    def test_opening_tag_stripped(self):
        raw = "<think>My reasoning here.</think>Final answer."
        cot, answer = split_thinking(raw, "think")
        assert not cot.startswith("<think>")
        assert cot == "My reasoning here."

    def test_whitespace_around_tags(self):
        raw = "  <think>  reasoning  </think>  answer  "
        cot, answer = split_thinking(raw, "think")
        assert cot == "reasoning"
        assert answer == "answer"

    def test_custom_tag_name(self):
        raw = "<reasoning>My CoT</reasoning>The answer."
        cot, answer = split_thinking(raw, "reasoning")
        assert cot == "My CoT"
        assert answer == "The answer."

    def test_multiple_closing_tags_splits_on_first(self):
        raw = "<think>part1</think>middle</think>end"
        cot, answer = split_thinking(raw, "think")
        assert cot == "part1"
        assert answer == "middle</think>end"

    def test_custom_cot_pattern(self):
        raw = "REASONING: I calculated. ANSWER: 42"
        pattern = r"REASONING:\s*(?P<cot>.*?)\s*ANSWER:\s*(?P<answer>.*)"
        cot, answer = split_thinking(raw, None, cot_pattern=pattern)
        assert cot == "I calculated."
        assert answer == "42"

    def test_custom_cot_pattern_no_match_no_tag(self):
        raw = "Just plain text"
        pattern = r"REASONING:\s*(?P<cot>.*?)\s*ANSWER:\s*(?P<answer>.*)"
        cot, answer = split_thinking(raw, None, cot_pattern=pattern)
        assert cot == ""
        assert answer == "Just plain text"

    def test_custom_cot_pattern_non_thinking(self):
        """Custom pattern used for non-thinking model when no JSON found."""
        raw = "REASONING: custom CoT ANSWER: custom answer"
        pattern = r"REASONING:\s*(?P<cot>.*?)\s*ANSWER:\s*(?P<answer>.*)"
        cot, answer = split_thinking(raw, None, cot_pattern=pattern)
        assert cot == "custom CoT"
        assert answer == "custom answer"

    def test_thinking_tag_takes_precedence_over_cot_pattern(self):
        """With thinking_tag set, tag-based splitting is used regardless of cot_pattern."""
        raw = "<think>My reasoning</think>My answer"
        pattern = r"REASONING:\s*(?P<cot>.*?)\s*ANSWER:\s*(?P<answer>.*)"
        cot, answer = split_thinking(raw, "think", cot_pattern=pattern)
        assert cot == "My reasoning"
        assert answer == "My answer"

    # --- JSON answer extraction ---

    def test_json_answer_non_thinking(self):
        """Non-thinking model outputs JSON answer."""
        raw = 'Step 1: compute.\nStep 2: done.\n{"final_answer": "42"}'
        cot, answer = split_thinking(raw, None)
        assert answer == "42"
        assert "Step 1" in cot

    def test_json_answer_thinking_model(self):
        """Thinking model outputs JSON in answer portion."""
        raw = '<think>Some reasoning</think>\n{"final_answer": "42"}'
        cot, answer = split_thinking(raw, "think")
        assert answer == "42"
        assert "Some reasoning" in cot

    def test_json_answer_with_escaped_quotes(self):
        """JSON answer containing escaped quotes."""
        raw = 'Reasoning here.\n{"final_answer": "He said \\"hello\\""}'
        cot, answer = split_thinking(raw, None)
        assert answer == 'He said "hello"'

    def test_json_answer_unquoted_number(self):
        """JSON answer with unquoted numeric value."""
        raw = 'Let me think...\n{"final_answer": 42}'
        cot, answer = split_thinking(raw, None)
        assert answer == "42"

    def test_json_answer_embedded_in_text(self):
        """JSON answer surrounded by extra text."""
        raw = 'I think the answer is:\n{"final_answer": "Paris"}\nHope that helps!'
        cot, answer = split_thinking(raw, None)
        assert answer == "Paris"

    def test_json_answer_multiline_value(self):
        """JSON answer with a value containing newlines (escaped)."""
        raw = 'Reasoning.\n{"final_answer": "line1\\nline2"}'
        cot, answer = split_thinking(raw, None)
        assert "line1" in answer
        assert "line2" in answer

    def test_no_json_falls_back_to_last_line(self):
        """No JSON found — falls back to cot_pattern (last line)."""
        from generation.model_registry import COT_LAST_LINE_PATTERN
        raw = "Step 1: Think.\nStep 2: Compute.\n42"
        cot, answer = split_thinking(raw, None, cot_pattern=COT_LAST_LINE_PATTERN)
        assert answer == "42"
        assert "Step 1" in cot

    def test_no_json_single_line(self):
        """Single line, no JSON — raw returned as answer."""
        raw = "42"
        cot, answer = split_thinking(raw, None)
        assert cot == ""
        assert answer == "42"

    def test_thinking_tag_no_json_plain_answer(self):
        """Thinking model, no JSON in answer — returns raw answer text."""
        raw = "<think>Reasoning</think>The answer is 42."
        cot, answer = split_thinking(raw, "think")
        assert "Reasoning" in cot
        assert answer == "The answer is 42."


# =========================================================================
# Config
# =========================================================================


class TestGenerationConfig:
    """Tests for config.py"""

    def test_save_and_load_roundtrip(self):
        config = GenerationConfig(
            model="Qwen/Qwen3-4B-Thinking-2507",
            seed=123,
            temperature=0.8,
            top_p=0.9,
            top_k=20,
            max_tokens=4096,
            thinking_tag="think",
            enable_thinking=True,
            input_path="/tmp/test.csv",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            config.output_path = str(Path(tmpdir) / "results.csv")
            path = config.save()
            assert path.name == "results_run_config.json"
            loaded = GenerationConfig.load(path)
            assert loaded.model == config.model
            assert loaded.seed == config.seed
            assert loaded.temperature == config.temperature
            assert loaded.top_p == config.top_p
            assert loaded.top_k == config.top_k
            assert loaded.max_tokens == config.max_tokens
            assert loaded.thinking_tag == config.thinking_tag
            assert loaded.enable_thinking == config.enable_thinking

    def test_save_creates_directory(self):
        config = GenerationConfig(model="test")
        with tempfile.TemporaryDirectory() as tmpdir:
            nested = Path(tmpdir) / "a" / "b" / "c" / "output.csv"
            config.output_path = str(nested)
            path = config.save()
            assert path.exists()

    def test_save_produces_valid_json(self):
        config = GenerationConfig(model="test", seed=42)
        with tempfile.TemporaryDirectory() as tmpdir:
            config.output_path = str(Path(tmpdir) / "results.csv")
            path = config.save()
            with open(path) as f:
                data = json.load(f)
            assert data["model"] == "test"
            assert data["seed"] == 42

    def test_config_json_named_after_output(self):
        config = GenerationConfig(model="test")
        with tempfile.TemporaryDirectory() as tmpdir:
            config.output_path = str(Path(tmpdir) / "Qwen3-4B_hle.csv")
            path = config.save()
            assert path.name == "Qwen3-4B_hle_run_config.json"

    def test_generate_run_id(self):
        config = GenerationConfig(model="Qwen/Qwen3-4B-Thinking-2507", seed=42)
        config.generate_run_id()
        assert "qwen3-4b-thinking-2507" in config.run_id
        assert "42" in config.run_id
        assert config.timestamp != ""

    def test_populate_input_hash(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("question\nWhat is 2+2?\n")
            f.flush()
            config = GenerationConfig(input_path=f.name)
            config.populate_input_hash()
            assert len(config.input_sha256) == 64  # SHA256 hex digest

    def test_populate_input_hash_same_content_same_hash(self):
        content = "question\nWhat is 2+2?\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f1:
            f1.write(content)
            f1.flush()
            c1 = GenerationConfig(input_path=f1.name)
            c1.populate_input_hash()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f2:
            f2.write(content)
            f2.flush()
            c2 = GenerationConfig(input_path=f2.name)
            c2.populate_input_hash()

        assert c1.input_sha256 == c2.input_sha256

    def test_populate_environment(self):
        config = GenerationConfig()
        config.populate_environment()
        assert config.python_version != ""
        assert config.torch_version != ""


# =========================================================================
# Backends helpers (no GPU required)
# =========================================================================


class TestRowSeed:
    """Tests for _row_seed"""

    def test_deterministic(self):
        assert _row_seed(42, 0) == 42
        assert _row_seed(42, 1) == 43
        assert _row_seed(42, 100) == 142

    def test_different_base_seeds(self):
        assert _row_seed(0, 5) != _row_seed(1, 5)

    def test_unique_per_row(self):
        seeds = [_row_seed(42, i) for i in range(1000)]
        assert len(set(seeds)) == 1000


class TestSetAllSeeds:
    """Tests for _set_all_seeds reproducibility"""

    def test_torch_reproducible(self):
        _set_all_seeds(42)
        a = torch.randn(100)
        _set_all_seeds(42)
        b = torch.randn(100)
        assert torch.equal(a, b)

    def test_numpy_reproducible(self):
        _set_all_seeds(42)
        a = np.random.rand(100)
        _set_all_seeds(42)
        b = np.random.rand(100)
        assert np.array_equal(a, b)

    def test_different_seeds_differ(self):
        _set_all_seeds(42)
        a = torch.randn(100)
        _set_all_seeds(99)
        b = torch.randn(100)
        assert not torch.equal(a, b)

    def test_python_random_reproducible(self):
        import random

        _set_all_seeds(42)
        a = [random.random() for _ in range(100)]
        _set_all_seeds(42)
        b = [random.random() for _ in range(100)]
        assert a == b


class TestBuildPrompt:
    """Tests for build_prompt"""

    class _MockTokenizer:
        """Mock tokenizer for testing prompt construction."""

        def apply_chat_template(self, messages, **kwargs):
            parts = []
            for m in messages:
                parts.append(f"<|{m['role']}|>{m['content']}")
            if kwargs.get("add_generation_prompt"):
                parts.append("<|assistant|>")
            extra = {
                k: v
                for k, v in kwargs.items()
                if k not in ("add_generation_prompt", "tokenize")
            }
            if extra:
                parts.append(f"[kwargs={extra}]")
            return "\n".join(parts)

    def test_basic_prompt(self):
        tok = self._MockTokenizer()
        prompt = build_prompt(tok, "What is 2+2?", None, False, {})
        assert "<|user|>What is 2+2?" in prompt
        assert "<|assistant|>" in prompt
        assert "<|system|>" not in prompt

    def test_with_system_message(self):
        tok = self._MockTokenizer()
        prompt = build_prompt(tok, "What is 2+2?", "You are helpful.", False, {})
        assert "<|system|>You are helpful." in prompt
        assert "<|user|>What is 2+2?" in prompt

    def test_enable_thinking(self):
        tok = self._MockTokenizer()
        prompt = build_prompt(tok, "Q?", None, True, {})
        assert "enable_thinking" in prompt

    def test_no_enable_thinking(self):
        tok = self._MockTokenizer()
        prompt = build_prompt(tok, "Q?", None, False, {})
        assert "enable_thinking" not in prompt

    def test_extra_chat_template_kwargs(self):
        tok = self._MockTokenizer()
        prompt = build_prompt(tok, "Q?", None, False, {"custom_key": "val"})
        assert "custom_key" in prompt

    def test_empty_system_message_excluded(self):
        tok = self._MockTokenizer()
        prompt = build_prompt(tok, "Q?", "", False, {})
        assert "<|system|>" not in prompt


# =========================================================================
# Generate helpers
# =========================================================================


class TestApplyProfileDefaults:
    """Tests for _apply_profile_defaults"""

    def test_profile_applied_when_no_overrides(self):
        config = GenerationConfig(model="Qwen/Qwen3-4B-Thinking-2507")
        profile = ModelProfile(
            temperature=0.6, top_p=0.95, top_k=20, max_tokens=32768,
            thinking_tag="think", enable_thinking=True, trust_remote_code=True,
        )
        _apply_profile_defaults(config, profile, cli_overrides=set())
        assert config.temperature == 0.6
        assert config.top_p == 0.95
        assert config.top_k == 20
        assert config.max_tokens == 32768
        assert config.thinking_tag == "think"
        assert config.enable_thinking is True
        assert config.trust_remote_code is True

    def test_cli_override_wins(self):
        config = GenerationConfig(model="Qwen/Qwen3-4B-Thinking-2507", temperature=0.8)
        profile = ModelProfile(temperature=0.6, top_p=0.95)
        _apply_profile_defaults(config, profile, cli_overrides={"temperature"})
        assert config.temperature == 0.8  # CLI wins
        assert config.top_p == 0.95  # Profile applied

    def test_multiple_overrides(self):
        config = GenerationConfig(
            model="test", temperature=0.9, top_k=50, dtype="float16"
        )
        profile = ModelProfile(
            temperature=0.6, top_p=0.95, top_k=20, dtype="bfloat16"
        )
        _apply_profile_defaults(
            config, profile, cli_overrides={"temperature", "top_k", "dtype"}
        )
        assert config.temperature == 0.9
        assert config.top_k == 50
        assert config.dtype == "float16"
        assert config.top_p == 0.95  # From profile

    def test_all_fields_overridden(self):
        """When everything is overridden, profile values are ignored."""
        config = GenerationConfig(
            model="test", temperature=0.9, top_p=0.8, top_k=50,
            max_tokens=1000, thinking_tag=None, enable_thinking=False,
            cot_pattern="custom", dtype="float16", trust_remote_code=True,
            max_model_len=2048,
        )
        profile = ModelProfile(temperature=0.1)  # Should be ignored
        all_fields = {
            "temperature", "top_p", "top_k", "max_tokens", "thinking_tag",
            "enable_thinking", "cot_pattern", "dtype", "trust_remote_code",
            "max_model_len", "chat_template_kwargs",
        }
        _apply_profile_defaults(config, profile, cli_overrides=all_fields)
        assert config.temperature == 0.9  # Unchanged


class TestDeriveOutputPath:
    """Tests for _derive_output_path"""

    def test_basic_derivation(self):
        config = GenerationConfig(model="Qwen/Qwen3-4B-Thinking-2507")
        df = pd.DataFrame({"question": ["Q1"], "dataset": ["hle"]})
        path = _derive_output_path(config, df)
        assert path == str(Path("faithbench") / "generation" / "Qwen3-4B-Thinking-2507_hle.csv")

    def test_different_dataset(self):
        config = GenerationConfig(model="meta-llama/Llama-3.3-70B-Instruct")
        df = pd.DataFrame({"question": ["Q1"], "dataset": ["gpqa"]})
        path = _derive_output_path(config, df)
        assert path == str(Path("faithbench") / "generation" / "Llama-3.3-70B-Instruct_gpqa.csv")

    def test_missing_dataset_column_raises(self):
        config = GenerationConfig(model="Qwen/Qwen3-4B-Thinking-2507")
        df = pd.DataFrame({"question": ["Q1"]})
        with pytest.raises(ValueError, match="missing required column 'dataset'"):
            _derive_output_path(config, df)

    def test_custom_dataset_column(self):
        config = GenerationConfig(model="Qwen/Qwen3-4B-Thinking-2507", dataset_column="benchmark")
        df = pd.DataFrame({"question": ["Q1"], "benchmark": ["mmlu"]})
        path = _derive_output_path(config, df)
        assert "Qwen3-4B-Thinking-2507_mmlu.csv" in path

    def test_uses_first_row_value(self):
        config = GenerationConfig(model="Qwen/Qwen3-4B-Thinking-2507")
        df = pd.DataFrame({"question": ["Q1", "Q2"], "dataset": ["hle", "hle"]})
        path = _derive_output_path(config, df)
        assert "hle" in path


class TestLoadInput:
    """Tests for _load_input"""

    def test_load_csv_with_prompt_column(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("id,prompt\n0,What is 2+2?\n1,What is 3+3?\n")
            f.flush()
            config = GenerationConfig(
                input_path=f.name, prompt_column="prompt", id_column="id"
            )
            df = _load_input(config)
            assert len(df) == 2
            assert config.num_rows == 2
            assert "prompt" in df.columns

    def test_missing_prompt_column_raises(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("id,text\n0,hello\n")
            f.flush()
            config = GenerationConfig(
                input_path=f.name, prompt_column="prompt", id_column="id"
            )
            with pytest.raises(ValueError, match="missing required column"):
                _load_input(config)

    def test_auto_generate_id_column(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("prompt\nWhat is 2+2?\nWhat is 3+3?\n")
            f.flush()
            config = GenerationConfig(
                input_path=f.name, prompt_column="prompt", id_column="id"
            )
            df = _load_input(config)
            assert "id" in df.columns
            assert list(df["id"]) == [0, 1]


class TestLoadCompletedIds:
    """Tests for _load_completed_ids"""

    def test_no_existing_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            ids = _load_completed_ids(Path(tmpdir) / "nonexistent.csv", "id")
            assert ids == set()

    def test_existing_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.csv"
            pd.DataFrame({"id": [0, 1, 2], "model_answer": ["a", "b", "c"]}).to_csv(
                output_path, index=False
            )
            ids = _load_completed_ids(output_path, "id")
            assert ids == {0, 1, 2}

    def test_missing_id_column_in_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.csv"
            pd.DataFrame({"model_answer": ["a", "b"]}).to_csv(output_path, index=False)
            ids = _load_completed_ids(output_path, "id")
            assert ids == set()


class TestResumeFlag:
    """Tests that --resume actually gates resume behavior."""

    def test_no_resume_with_existing_results_raises(self):
        """Without --resume, existing output file should raise FileExistsError."""
        from generation.generate import run

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.csv"
            output_path.write_text("id,model_answer\n0,test\n")

            # Create input CSV
            input_path = Path(tmpdir) / "input.csv"
            input_path.write_text("id,prompt,dataset\n0,What is 2+2?,hle\n")

            config = GenerationConfig(
                model="Qwen/Qwen3-4B-Thinking-2507",
                input_path=str(input_path),
                output_path=str(output_path),
                resume=False,
            )
            config.generate_run_id()

            with pytest.raises(FileExistsError, match="Use --resume"):
                run(config, cli_overrides=set())

    def test_config_resume_defaults_false(self):
        """Resume should default to False."""
        config = GenerationConfig()
        assert config.resume is False


class TestEnableThinkingValidation:
    """Tests that enable_thinking is validated against model type."""

    def test_enable_thinking_on_instruct_model_raises(self):
        """Passing --enable_thinking with an Instruct model should raise."""
        from generation.generate import run

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.csv"
            input_path.write_text("id,prompt,dataset\n0,What is 2+2?,hle\n")

            config = GenerationConfig(
                model="Qwen/Qwen3-4B-Instruct-2507",
                input_path=str(input_path),
                output_path=str(Path(tmpdir) / "output.csv"),
                enable_thinking=True,  # Conflicts with Instruct model
            )
            config.generate_run_id()

            with pytest.raises(ValueError, match="--enable_thinking conflicts"):
                run(config, cli_overrides={"enable_thinking"})

    def test_enable_thinking_on_thinking_model_ok(self):
        """Passing --enable_thinking with a Thinking model should not raise
        (it matches the profile, so no conflict)."""
        from generation.generate import run

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.csv"
            input_path.write_text("id,prompt,dataset\n0,What is 2+2?,hle\n")

            config = GenerationConfig(
                model="Qwen/Qwen3-4B-Thinking-2507",
                input_path=str(input_path),
                output_path=str(Path(tmpdir) / "output.csv"),
                enable_thinking=True,
            )
            config.generate_run_id()

            # Should not raise ValueError for enable_thinking
            # (will fail later at backend init, but that's expected)
            with pytest.raises(Exception) as exc_info:
                run(config, cli_overrides={"enable_thinking"})
            # It should NOT be a ValueError about enable_thinking
            assert "enable_thinking conflicts" not in str(exc_info.value)


class TestSaveResults:
    """Tests for _save_results"""

    def test_save_fresh(self):
        rows = [
            {"id": 0, "model_answer": "4", "cot": "reasoning"},
            {"id": 1, "model_answer": "6", "cot": "more reasoning"},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.csv"
            _save_results(rows, output_path, append=False)
            result = pd.read_csv(output_path)
            assert len(result) == 2
            assert list(result["id"]) == [0, 1]
            assert "model_answer" in result.columns

    def test_save_append(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.csv"
            rows1 = [{"id": 0, "model_answer": "4"}]
            _save_results(rows1, output_path, append=False)

            rows2 = [{"id": 1, "model_answer": "6"}]
            _save_results(rows2, output_path, append=True)

            result = pd.read_csv(output_path)
            assert len(result) == 2

    def test_save_empty_rows_noop(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.csv"
            _save_results([], output_path, append=False)
            assert not output_path.exists()

    def test_csv_content_integrity(self):
        repro = json.dumps({"seed_used": 42, "raw_output": "line1\nline2"})
        rows = [{"id": 0, "cot": 'has "quotes"', "model_answer": "4", "reproducibility_info": repro}]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.csv"
            _save_results(rows, output_path, append=False)
            result = pd.read_csv(output_path)
            assert result.iloc[0]["cot"] == 'has "quotes"'
            # Verify reproducibility_info is valid JSON
            info = json.loads(result.iloc[0]["reproducibility_info"])
            assert info["seed_used"] == 42
            assert info["raw_output"] == "line1\nline2"

    def test_output_preserves_input_columns(self):
        """Output CSV should contain all input columns plus generation columns."""
        rows = [
            {
                "id": 0, "question": "What is 2+2?", "dataset": "hle",
                "extra_field": "some_value",
                "cot": "reasoning", "model_answer": "4",
                "reproducibility_info": "{}",
            }
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results.csv"
            _save_results(rows, output_path, append=False)
            result = pd.read_csv(output_path)
            assert "question" in result.columns
            assert "dataset" in result.columns
            assert "extra_field" in result.columns
            assert "cot" in result.columns
            assert "model_answer" in result.columns
            assert "reproducibility_info" in result.columns


# =========================================================================
# GenerationResult
# =========================================================================


class TestGenerationResult:
    """Tests for GenerationResult dataclass"""

    def test_creation(self):
        r = GenerationResult(
            row_index=0, raw_output="test", finish_reason="stop",
            num_tokens=10, seed_used=42,
        )
        assert r.row_index == 0
        assert r.raw_output == "test"
        assert r.finish_reason == "stop"
        assert r.num_tokens == 10
        assert r.seed_used == 42

    def test_no_error_field(self):
        """GenerationResult should NOT have an error field — errors must raise."""
        assert not hasattr(
            GenerationResult(
                row_index=0, raw_output="", finish_reason="stop",
                num_tokens=0, seed_used=0,
            ),
            "error",
        )


# =========================================================================
# Integration-style tests (no GPU)
# =========================================================================


class TestProfileConfigIntegration:
    """Test that model profiles correctly flow into config."""

    def test_qwen3_thinking_profile_to_config(self):
        config = GenerationConfig(model="Qwen/Qwen3-4B-Thinking-2507")
        profile = get_model_profile("Qwen/Qwen3-4B-Thinking-2507")
        _apply_profile_defaults(config, profile, cli_overrides=set())

        assert config.temperature == 0.6
        assert config.top_p == 0.95
        assert config.top_k == 20
        assert config.max_tokens == 32768
        assert config.thinking_tag == "think"
        assert config.enable_thinking is True
        assert config.trust_remote_code is True

    def test_qwen3_instruct_profile_to_config(self):
        config = GenerationConfig(model="Qwen/Qwen3-4B-Instruct-2507")
        profile = get_model_profile("Qwen/Qwen3-4B-Instruct-2507")
        _apply_profile_defaults(config, profile, cli_overrides=set())

        assert config.temperature == 0.7
        assert config.top_p == 0.8
        assert config.top_k == 20
        assert config.max_tokens == 16384
        assert config.thinking_tag is None
        assert config.enable_thinking is False

    def test_llama_profile_to_config(self):
        config = GenerationConfig(model="meta-llama/Llama-3.3-70B-Instruct")
        profile = get_model_profile("meta-llama/Llama-3.3-70B-Instruct")
        _apply_profile_defaults(config, profile, cli_overrides=set())

        assert config.temperature == 0.7
        assert config.top_p == 0.9
        assert config.thinking_tag is None
        assert config.enable_thinking is False

    def test_olmo_think_profile_to_config(self):
        config = GenerationConfig(model="allenai/Olmo-3-7B-Think")
        profile = get_model_profile("allenai/Olmo-3-7B-Think")
        _apply_profile_defaults(config, profile, cli_overrides=set())

        assert config.thinking_tag == "think"
        assert config.enable_thinking is False  # OLMo doesn't use this kwarg
        assert config.max_tokens == 32768
        assert config.top_k == 50

    def test_deepseek_r1_distill_profile_to_config(self):
        config = GenerationConfig(model="deepseek-ai/DeepSeek-R1-Distill-Llama-70B")
        profile = get_model_profile("deepseek-ai/DeepSeek-R1-Distill-Llama-70B")
        _apply_profile_defaults(config, profile, cli_overrides=set())

        assert config.thinking_tag == "think"
        assert config.enable_thinking is False
        assert config.max_tokens == 32768

    def test_cli_override_with_real_profile(self):
        """Simulate: user passes --temperature 0.8 with Qwen3 Thinking model."""
        config = GenerationConfig(model="Qwen/Qwen3-4B-Thinking-2507", temperature=0.8)
        profile = get_model_profile("Qwen/Qwen3-4B-Thinking-2507")
        _apply_profile_defaults(config, profile, cli_overrides={"temperature"})

        assert config.temperature == 0.8  # CLI override
        assert config.top_k == 20  # From profile
        assert config.enable_thinking is True  # From profile
