"""CoT segmentation, POS tagging, and token-word alignment.

Segments a chain-of-thought into sentences, identifies content words via SpaCy
POS tagging, and aligns them to HuggingFace subword token spans.

Origin: Copied from https://github.com/technion-cs-nlp/parametric-faithfulness/blob/main/segment.py
Changes:
  - Added StepInfo dataclass for richer output
  - Added detect_whitespace_char() to auto-detect instead of hardcoded dict
  - Added segment_and_filter() convenience function
  - Added lazy SpaCy loading via get_nlp()
  - Wrapped in module with docstrings
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Content word POS tags — from reference segment.py
TARGET_TAGS = {"VERB", "NUM", "ADJ", "NOUN", "PROPN"}

# Lazy-loaded SpaCy model
_nlp = None


def get_nlp():
    """Lazy-load SpaCy en_core_web_sm model."""
    global _nlp
    if _nlp is None:
        import spacy

        _nlp = spacy.load("en_core_web_sm", disable=["ner"])
    return _nlp


@dataclass
class Word:
    """A word with its POS tag and token span indices.

    Origin: From reference segment.py, unchanged.
    """

    word: str
    pos: str
    span_start: int  # inclusive, index into token list
    span_end: int  # exclusive, index into token list

    def is_content(self) -> bool:
        """Whether this word is a content word (noun, verb, adj, num, propn)."""
        return self.pos in TARGET_TAGS


@dataclass
class StepInfo:
    """Information about a single CoT step after segmentation and filtering.

    New addition — not in reference code.
    """

    text: str  # The sentence text
    char_start: int  # Start offset in the original CoT string
    char_end: int  # End offset in the original CoT string
    token_ids: torch.Tensor  # Token IDs for this step
    content_words: list[Word]  # Content words with token spans


def detect_whitespace_char(tokenizer) -> str:
    """Auto-detect the whitespace marker character used by the tokenizer.

    New addition — replaces hardcoded WHITESPACE_CHARS dict from reference.

    GPT-2/Llama-style tokenizers use 'Ġ', SentencePiece-based use '▁'.
    """
    tokens = tokenizer.tokenize(" test")
    if tokens and tokens[0].startswith("Ġ"):
        return "Ġ"
    if tokens and tokens[0].startswith("▁"):
        return "▁"
    # Fallback: try encoding and checking
    token_strs = tokenizer.convert_ids_to_tokens(
        tokenizer.encode(" test", add_special_tokens=False)
    )
    for t in token_strs:
        if t.startswith("Ġ"):
            return "Ġ"
        if t.startswith("▁"):
            return "▁"
    # Default to Ġ
    return "Ġ"


def sentencize(text: str) -> list[str]:
    """Split text into sentences using NLTK.

    Origin: From reference segment.py, unchanged.
    """
    import nltk

    try:
        return nltk.sent_tokenize(text)
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
        return nltk.sent_tokenize(text)


def pos_tag(text: str, nlp=None) -> list[tuple[str, str]]:
    """POS-tag words in text using SpaCy.

    Origin: From reference segment.py, unchanged.

    Returns:
        List of (word_text, pos_tag) tuples.
    """
    if nlp is None:
        nlp = get_nlp()
    doc = nlp(text)
    return [(w.text, w.pos_) for w in doc]


def words_to_token_spans(
    wpos: list[tuple[str, str]], tokens: list[str], whitespace_char: str
) -> list[Word]:
    """Align SpaCy words to HuggingFace subword token spans.

    Origin: From reference segment.py, unchanged logic.
    Changes: Added type hints and docstring.

    Args:
        wpos: List of (word, pos) from SpaCy.
        tokens: Subword tokens from tokenizer.tokenize().
        whitespace_char: The whitespace marker char ('Ġ' or '▁').

    Returns:
        List of Word objects with token span indices.
    """
    # Filter out space tokens
    toks_pos = [(t, p) for t, p in wpos if p != "SPACE"]

    if not toks_pos:
        return []

    i = 0
    cur_word, cur_pos = toks_pos[i]

    word_start = 0
    words = []

    for j, subword in enumerate(tokens):
        if whitespace_char in subword:  # new word
            word_start = j

        # Convert span to string, filter out whitespace
        span = tokens[word_start : j + 1]
        span = [e.replace(whitespace_char, "") for e in span]
        cur = "".join(span)

        # equality check
        if cur == cur_word:
            w = Word(cur_word, cur_pos, word_start, j + 1)
            words.append(w)

            i += 1
            if i >= len(toks_pos):
                break

            cur_word, cur_pos = toks_pos[i]
            word_start = j

    return words


def align_cot_to_pos(
    cot_step_text: str,
    tokenizer,
    whitespace_char: str,
    nlp=None,
) -> tuple[torch.Tensor, list[Word]]:
    """Tokenize a CoT step and align words to token spans with POS tags.

    Origin: From reference segment.py.
    Changes: Takes whitespace_char directly instead of looking up model_id in a dict.

    Args:
        cot_step_text: The text of one CoT step (sentence).
        tokenizer: HuggingFace tokenizer.
        whitespace_char: Whitespace marker char for this tokenizer.
        nlp: SpaCy model (lazy-loaded if None).

    Returns:
        (token_ids, words): Token ID tensor and list of Word objects.
    """
    if nlp is None:
        nlp = get_nlp()

    w_p = pos_tag(cot_step_text, nlp)
    pretokenized_text = [f" {w}" for w, _ in w_p]  # Prefix whitespace per word

    # Some tokenizers (e.g. GPT2-based, used by OLMo) require add_prefix_space=True
    # for is_split_into_words=True. Fall back to tokenizing the joined string.
    try:
        tokens = tokenizer.tokenize(
            pretokenized_text, is_split_into_words=True, add_special_tokens=False
        )
    except (AssertionError, Exception):
        tokens = tokenizer.tokenize(
            "".join(pretokenized_text), add_special_tokens=False
        )

    indices = torch.tensor(tokenizer.convert_tokens_to_ids(tokens))

    return indices, words_to_token_spans(w_p, tokens, whitespace_char)


def segment_and_filter(
    cot: str,
    tokenizer,
    min_content_tokens: int = 2,
    nlp=None,
) -> list[StepInfo]:
    """Segment a CoT into steps, POS-tag, and filter by content word count.

    New addition — convenience function combining sentencize + align + filter.

    Args:
        cot: Full chain-of-thought text.
        tokenizer: HuggingFace tokenizer.
        min_content_tokens: Minimum content words required to keep a step.
        nlp: SpaCy model (lazy-loaded if None).

    Returns:
        List of StepInfo for steps that pass the content word filter.
    """
    if nlp is None:
        nlp = get_nlp()

    from sentence_splitting import split_into_sentences

    whitespace_char = detect_whitespace_char(tokenizer)
    sentences = split_into_sentences(cot)

    steps = []
    offset = 0
    for sent in sentences:
        # Find the sentence in the original CoT text
        idx = cot.find(sent, offset)
        if idx == -1:
            # Fallback: use current offset
            idx = offset
        char_start = idx
        char_end = idx + len(sent)
        offset = char_end

        # Tokenize and align
        token_ids, words = align_cot_to_pos(sent, tokenizer, whitespace_char, nlp)

        # Filter: keep only content words
        content_words = [w for w in words if w.is_content()]

        # Skip steps with too few content tokens
        if len(content_words) < min_content_tokens:
            continue

        steps.append(
            StepInfo(
                text=sent,
                char_start=char_start,
                char_end=char_end,
                token_ids=token_ids,
                content_words=content_words,
            )
        )

    return steps
