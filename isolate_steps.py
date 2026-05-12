import re
import asyncio
import json
import threading
from datetime import datetime

import torch
from torch.nn import functional as F
from tqdm import tqdm
from google import genai
from google.genai import types
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import CrossEncoder

from env import GEMINI_API_KEY
from sentence_splitting import split_into_sentences  # re-exported for callers


# =============================================================================
# CONFIGURATION: Choose which entailment model to use
# =============================================================================
# Options:
#   "t5"      - google/t5_xxl_true_nli_mixture (larger, slower, but high quality)
#   "deberta" - MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli (faster)
#   "finecat" - dleemiller/finecat-nli-l (fast, lightweight)
ENTAILMENT_MODEL = "t5"


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# Lazy loading for entailment models
# =============================================================================

_t5_nli_tokenizer = None
_t5_nli_model = None
_crossencoder_nli_model = None

_entailment_lock = threading.Lock()


def get_entailment_model():
    """Lazy load the entailment model based on ENTAILMENT_MODEL config.

    Must be called while holding _entailment_lock (the scoring functions
    acquire it and keep it held through both loading and inference).
    """
    global _t5_nli_tokenizer, _t5_nli_model, _crossencoder_nli_model

    if ENTAILMENT_MODEL == "t5":
        if _t5_nli_tokenizer is None:
            print(f"Loading entailment model: {ENTAILMENT_MODEL}")
            _t5_nli_tokenizer = AutoTokenizer.from_pretrained("google/t5_xxl_true_nli_mixture")
            _t5_nli_model = AutoModelForSeq2SeqLM.from_pretrained(
                "google/t5_xxl_true_nli_mixture", device_map=_get_device(), torch_dtype=torch.bfloat16
            )
            _t5_nli_model = _t5_nli_model.eval()
        return _t5_nli_tokenizer, _t5_nli_model

    elif ENTAILMENT_MODEL == "deberta":
        if _crossencoder_nli_model is None:
            print(f"Loading entailment model: {ENTAILMENT_MODEL}")
            _crossencoder_nli_model = CrossEncoder("MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
        return _crossencoder_nli_model

    elif ENTAILMENT_MODEL == "finecat":
        if _crossencoder_nli_model is None:
            print(f"Loading entailment model: {ENTAILMENT_MODEL}")
            _crossencoder_nli_model = CrossEncoder("dleemiller/finecat-nli-l")
        return _crossencoder_nli_model

    else:
        raise ValueError(f"Unknown entailment model: {ENTAILMENT_MODEL}. Choose 't5', 'deberta', or 'finecat'.")


def _extract_retry_wait(error, default: float = 4.0) -> float:
    """Extract retry wait time from a Gemini API error.

    Checks Retry-After header and error message for suggested wait times.
    Falls back to `default` seconds if no hint is found.
    """
    resp = getattr(error, "response", None)
    if resp is not None:
        headers = getattr(resp, "headers", {})
        retry_after = headers.get("Retry-After") or headers.get("retry-after")
        if retry_after:
            try:
                return float(retry_after)
            except (ValueError, TypeError):
                pass

    msg = getattr(error, "message", "") or str(error)
    match = re.search(r"retry\s+(?:after|in)\s+(\d+(?:\.\d+)?)\s*s", msg, re.IGNORECASE)
    if match:
        return float(match.group(1))

    return default


class Judge:
    API_COSTS = {
        "gemini-3-flash-preview": {"input": 0.5 / 1000000, "output": 3.0 / 1000000},
        "gemini-3-pro-preview": {"input": 2.0 / 1000000, "output": 12.0 / 1000000},
        "gemini-3.1-pro-preview": {"input": 2.0 / 1000000, "output": 12.0 / 1000000},
        "gemini-3.1-flash-lite-preview": {"input": 0.25 / 1000000, "output": 1.5 / 1000000},
    }

    def __init__(self, model_name="gemini-3-flash-preview", api_key=GEMINI_API_KEY):
        print(f"Initializing Judge with model {model_name}")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.token_count = 0
        self.total_cost = 0.0

    def _build_config(self, max_output_tokens: int | None = None) -> types.GenerateContentConfig:
        """Build a GenerateContentConfig, optionally with max_output_tokens."""
        kwargs = dict(
            thinking_config=types.ThinkingConfig(include_thoughts=True),
        )
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens
        return types.GenerateContentConfig(**kwargs)

    def run(self, prompt, max_output_tokens: int | None = None):
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self._build_config(max_output_tokens),
        )
        self.token_count += response.usage_metadata.total_token_count
        self.total_cost += self.get_request_cost(response)
        return response

    def run_batch(self, prompts: list[str], max_concurrency: int = 10, use_tqdm: bool = False, max_output_tokens: int | None = None) -> list:
        """
        Run multiple prompts concurrently for faster processing.
        Uses async internally but presents a synchronous API.
        """
        if not prompts:
            return []

        if len(prompts) == 1:
            return [self.run(prompts[0], max_output_tokens=max_output_tokens)]

        pbar = tqdm(total=len(prompts)) if use_tqdm else None
        config = self._build_config(max_output_tokens)

        async def _run_batch():
            semaphore = asyncio.Semaphore(max_concurrency)

            async def run_one(prompt):
                async with semaphore:
                    max_retries = 5
                    for attempt in range(max_retries):
                        try:
                            response = await self.client.aio.models.generate_content(
                                model=self.model_name,
                                contents=prompt,
                                config=config,
                            )
                            if pbar is not None:
                                pbar.update(1)
                            return response
                        except Exception as e:
                            is_transient = getattr(e, "code", 0) in (429, 500, 503)
                            if is_transient and attempt < max_retries - 1:
                                wait = _extract_retry_wait(e, default=4 * 2 ** attempt)
                                print(f"[Judge] Transient error (attempt {attempt + 1}/{max_retries}): "
                                      f"{str(e)[:120]}. Retrying in {wait:.0f}s...")
                                await asyncio.sleep(wait)
                            else:
                                raise

            return await asyncio.gather(*[run_one(p) for p in prompts])

        try:
            loop = asyncio.get_running_loop()
            import nest_asyncio

            nest_asyncio.apply()
            responses = loop.run_until_complete(_run_batch())
        except RuntimeError:
            responses = asyncio.run(_run_batch())

        for resp in responses:
            self.token_count += resp.usage_metadata.total_token_count
            self.total_cost += self.get_request_cost(resp)

        return responses

    def get_thoughts(self, response):
        thoughts = []
        for candidate in response.candidates:
            for part in candidate.content.parts:
                if not part.text:
                    continue
                if part.thought:
                    thoughts.append(part.text)

        return "\n\n".join(thoughts)

    def get_token_counts(self, response):
        return (
            response.usage_metadata.thoughts_token_count,
            response.usage_metadata.candidates_token_count,
            response.usage_metadata.prompt_token_count,
        )

    def get_request_cost(self, response):
        thought_count, output_count, input_count = self.get_token_counts(response)
        output_count = 0 if output_count is None else output_count
        thought_count = 0 if thought_count is None else thought_count
        input_count = 0 if input_count is None else input_count
        return self.get_cost(input_count, output_count + thought_count)

    def get_cost(self, input_tokens, output_tokens):
        return (
            input_tokens * self.API_COSTS[self.model_name]["input"]
            + output_tokens * self.API_COSTS[self.model_name]["output"]
        )

    def reset_cost(self) -> float:
        """Return accumulated cost since last reset, then reset to 0."""
        cost = self.total_cost
        self.total_cost = 0.0
        return cost

    def close(self):
        with open(f"judge_token_usage_{self.model_name}.txt", "a+") as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{timestamp} {self.token_count}\n")


def try_really_hard_to_parse_json(output: str):
    """Try multiple strategies to parse JSON from LLM output that may contain extra text or formatting."""

    # Strategy 1: Try direct parsing
    try:
        return json.loads(output)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Strip whitespace and try again
    try:
        return json.loads(output.strip())
    except json.JSONDecodeError:
        pass

    # Strategy 3: Look for JSON between triple backticks (markdown code blocks)
    if "```" in output:
        try:
            start = output.find("```")
            end = output.rfind("```")
            if start != end:
                json_str = output[start:end].replace("```json", "").replace("```", "").strip()
                return json.loads(json_str)
        except json.JSONDecodeError:
            pass

    # Strategy 4: Find the first { and last } and try to parse that
    try:
        start = output.find("{")
        end = output.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = output[start : end + 1]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Strategy 5: Try to find JSON array [...]
    try:
        start = output.find("[")
        end = output.rfind("]")
        if start != -1 and end != -1 and end > start:
            json_str = output[start : end + 1]
            return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Strategy 6: Fix invalid backslash escapes (e.g. LaTeX: \cdot, \frac, \times)
    # In JSON, only \", \\, \/, \b, \f, \n, \r, \t, \uXXXX are valid escapes.
    def _fix_backslash_escapes(s):
        return re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', s)

    for extract_fn in [
        lambda s: s,
        lambda s: s[s.find("{"):s.rfind("}") + 1] if "{" in s and "}" in s else None,
        lambda s: s[s.find("["):s.rfind("]") + 1] if "[" in s and "]" in s else None,
    ]:
        try:
            extracted = extract_fn(output.strip().replace("```json", "").replace("```", "").strip())
            if extracted:
                return json.loads(_fix_backslash_escapes(extracted))
        except (json.JSONDecodeError, TypeError):
            pass

    print(f"Could not parse JSON from output after trying all strategies: {output}")
    raise json.JSONDecodeError(f"Could not parse JSON from output after trying all strategies", output, 0)


def get_top_step_candidates_using_entailment(cot_steps, step, k=None, batch_size=10, threshold=0.5):
    """
    Get candidate steps that entail a given hypothesis step.
    Uses the configured entailment model (T5 or DeBERTa).
    """
    cot_steps_list = list(cot_steps)

    if ENTAILMENT_MODEL == "t5":
        entailment_scores = _get_entailment_scores_t5(cot_steps_list, step, batch_size)
    elif ENTAILMENT_MODEL in ("deberta", "finecat"):
        entailment_scores = _get_entailment_scores_crossencoder(cot_steps_list, step, batch_size)
    else:
        raise ValueError(f"Unknown entailment model: {ENTAILMENT_MODEL}")

    indices_above_threshold = [i for i, score in enumerate(entailment_scores) if score >= threshold]

    if k is not None:
        topk_indices = sorted(range(len(entailment_scores)), key=lambda i: entailment_scores[i], reverse=True)[:k]
        topk_steps = [cot_steps_list[i] for i in topk_indices]
        topk_scores = [entailment_scores[i] for i in topk_indices]
        return topk_steps, topk_scores, topk_indices
    else:
        sorted_indices = sorted(indices_above_threshold, key=lambda i: entailment_scores[i], reverse=True)
        topk_steps = [cot_steps_list[i] for i in sorted_indices]
        topk_scores = [entailment_scores[i] for i in sorted_indices]
        return topk_steps, topk_scores, sorted_indices


def _get_entailment_scores_t5(cot_steps_list, step, batch_size):
    """Get entailment scores using T5 NLI model.

    Thread-safe: all GPU inference is serialized via _entailment_lock to
    prevent concurrent access to the model from multiple judge workers.
    """
    with _entailment_lock:
        t5_nli_tokenizer, t5_nli_model = get_entailment_model()

        entailment_scores = []
        entailment_token_id = t5_nli_tokenizer.encode("1", add_special_tokens=False)[0]

        for start in range(0, len(cot_steps_list), batch_size):
            batch_steps = cot_steps_list[start : start + batch_size]

            input_texts = [f"premise: {cot_step} hypothesis: {step}" for cot_step in batch_steps]

            inputs = t5_nli_tokenizer(
                input_texts,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
            ).to(_get_device())

            with torch.no_grad():
                outputs = t5_nli_model.generate(
                    **inputs,
                    max_length=10,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

                if len(outputs.scores) > 0:
                    first_token_logits = outputs.scores[0]
                    probs = F.softmax(first_token_logits, dim=-1)
                    batch_scores = probs[:, entailment_token_id]
                    entailment_scores.extend(batch_scores.detach().cpu().tolist())
                else:
                    preds = t5_nli_tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
                    entailment_scores.extend([1.0 if p.strip() == "1" else 0.0 for p in preds])

    return entailment_scores


def _get_entailment_scores_crossencoder(cot_steps_list, step, batch_size):
    """
    Get entailment scores using a CrossEncoder NLI model (DeBERTa or FineCat).
    The model outputs logits for [contradiction, neutral, entailment].
    We convert to probabilities and return the entailment probability.

    Thread-safe: all GPU inference is serialized via _entailment_lock.
    """
    with _entailment_lock:
        crossencoder_nli_model = get_entailment_model()

        entailment_scores = []

        for start in range(0, len(cot_steps_list), batch_size):
            batch_steps = cot_steps_list[start : start + batch_size]

            pairs = [(cot_step, step) for cot_step in batch_steps]

            logits = crossencoder_nli_model.predict(pairs)

            probs = F.softmax(torch.tensor(logits), dim=-1)
            entailment_probs = probs[:, 2].tolist()

            entailment_scores.extend(entailment_probs)

    return entailment_scores
