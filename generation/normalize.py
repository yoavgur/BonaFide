"""Normalize model answers for robust comparison.

Handles LaTeX formatting, Unicode math symbols, whitespace differences,
and common model output quirks so that semantically equivalent answers match.
"""

from __future__ import annotations

import re


# Unicode -> LaTeX replacements
_UNICODE_TO_LATEX = {
    "\u03c0": "\\pi",       # π
    "\u03c4": "\\tau",       # τ
    "\u03c9": "\\omega",     # ω
    "\u03b1": "\\alpha",     # α
    "\u03b2": "\\beta",      # β
    "\u03b3": "\\gamma",     # γ
    "\u03b4": "\\delta",     # δ
    "\u03b5": "\\epsilon",   # ε
    "\u03b8": "\\theta",     # θ
    "\u03bb": "\\lambda",    # λ
    "\u03bc": "\\mu",        # μ
    "\u03c3": "\\sigma",     # σ
    "\u03c6": "\\phi",       # φ
    "\u03c8": "\\psi",       # ψ
    "\u2208": "\\in",        # ∈
    "\u221e": "\\infty",     # ∞
    "\u00d7": "\\times",     # ×
    "\u00b7": "\\cdot",      # ·
    "\u00b1": "\\pm",        # ±
    "\u2264": "\\leq",       # ≤
    "\u2265": "\\geq",       # ≥
    "\u2260": "\\neq",       # ≠
    "\u2192": "\\to",        # →
    "\u2282": "\\subset",    # ⊂
    "\u2211": "\\sum",       # ∑
    "\u220f": "\\prod",      # ∏
    "\u222b": "\\int",       # ∫
    "\u221a": "\\sqrt",      # √
}

# Unicode superscript -> normal digits
_SUPERSCRIPTS = {
    "\u2070": "0", "\u00b9": "1", "\u00b2": "2", "\u00b3": "3",
    "\u2074": "4", "\u2075": "5", "\u2076": "6", "\u2077": "7",
    "\u2078": "8", "\u2079": "9", "\u207f": "n",
}

# Unicode subscript -> normal digits
_SUBSCRIPTS = {
    "\u2080": "0", "\u2081": "1", "\u2082": "2", "\u2083": "3",
    "\u2084": "4", "\u2085": "5", "\u2086": "6", "\u2087": "7",
    "\u2088": "8", "\u2089": "9",
}

# Bare Greek letter words -> LaTeX (applied after lowercasing)
_BARE_GREEK = {
    "alpha": "\\alpha", "beta": "\\beta", "gamma": "\\gamma",
    "delta": "\\delta", "epsilon": "\\epsilon", "theta": "\\theta",
    "lambda": "\\lambda", "mu": "\\mu", "sigma": "\\sigma",
    "omega": "\\omega", "phi": "\\phi", "psi": "\\psi",
    "pi": "\\pi", "tau": "\\tau", "infty": "\\infty",
}

# Prefixes that models add before the actual answer
_ANSWER_PREFIXES = re.compile(
    r"^(?:"
    r"(?:final|the)\s+answer\s*(?:is)?[:\s]*"
    r"|thus,?\s+(?:the\s+)?answer\s+is\s+"
    r"|therefore,?\s+(?:the\s+)?answer\s+is\s+"
    r"|answer[:\s]+"
    r")",
    re.IGNORECASE,
)


def normalize_answer(s: str) -> str:
    """Normalize an answer string for comparison.

    Applies a series of transformations to strip formatting differences
    so that semantically identical answers compare equal.

    Args:
        s: Raw answer string (model_answer or wrong_answer).

    Returns:
        Normalized string suitable for equality comparison.
    """
    if not s or not isinstance(s, str):
        return ""

    s = s.strip()

    # Strip common answer prefixes ("Final Answer:", "Thus, the answer is", etc.)
    s = _ANSWER_PREFIXES.sub("", s).strip()

    # Remove \boxed{...} — handle nested braces by finding matching close
    s = _remove_boxed(s)

    # Remove LaTeX delimiters: $...$, $$...$$, \(...\), \[...\]
    s = re.sub(r"^\$+\s*", "", s)
    s = re.sub(r"\s*\$+$", "", s)
    s = re.sub(r"\\\(|\\\)", "", s)
    s = re.sub(r"\\\[|\\]", "", s)

    # \dfrac -> \frac, then \frac{a}{b} -> a/b
    s = s.replace("\\dfrac", "\\frac")
    s = _frac_to_slash(s)

    # \exp(...) -> exp(...)
    s = s.replace("\\exp", "exp")

    # Remove \left and \right (don't change meaning)
    s = s.replace("\\left", "").replace("\\right", "")

    # Normalize bracket types: [ ] -> ( ) since they're interchangeable delimiters
    # in LaTeX math (e.g. \left[...\right] vs \left(...\right))
    s = s.replace("[", "(").replace("]", ")")

    # \text{X} -> X
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)

    # Remove \, \; \! \quad \qquad (LaTeX spacing commands)
    s = re.sub(r"\\[,;!\s]", "", s)
    s = re.sub(r"\\q(?:uad)+", "", s)

    # Unicode math symbols -> LaTeX
    for u, l in _UNICODE_TO_LATEX.items():
        s = s.replace(u, l)

    # Unicode superscripts -> ^digit
    for u, d in _SUPERSCRIPTS.items():
        s = s.replace(u, "^" + d)

    # Unicode subscripts -> _digit
    for u, d in _SUBSCRIPTS.items():
        s = s.replace(u, "_" + d)

    # Normalize multiplication: \cdot, \times, * -> *
    s = s.replace("\\cdot", "*").replace("\\times", "*")

    # Remove ALL whitespace
    s = re.sub(r"\s+", "", s)

    # Lowercase
    s = s.lower()

    # Strip outer set braces: {X} -> X (but not {a}{b} or nested)
    if s.startswith("{") and s.endswith("}") and s.count("{") == 1:
        s = s[1:-1]

    # Bare Greek words -> LaTeX (after lowercasing): omega -> \omega, etc.
    # Only replace whole words to avoid mangling substrings
    for word, latex in _BARE_GREEK.items():
        s = re.sub(rf"(?<![\\a-z]){word}(?![a-z])", re.escape(latex), s)

    # Numeric normalization: remove commas in numbers, strip trailing .0
    # "26,000" -> "26000", "16.0" -> "16"
    s = re.sub(r"(?<=\d),(?=\d{3})", "", s)  # thousand separators
    s = re.sub(r"\.0+$", "", s)  # trailing .0

    return s


def _frac_to_slash(s: str) -> str:
    r"""Convert \frac{a}{b} -> (a)/(b), handling nested braces."""
    while True:
        m = re.search(r"\\frac\{", s)
        if not m:
            break
        # Find numerator: matched braces after \frac{
        num_start = m.end()
        num_end = _find_matching_brace(s, m.end() - 1)
        if num_end is None:
            break
        numerator = s[num_start:num_end]

        # Find denominator: should be {..} right after
        if num_end + 1 >= len(s) or s[num_end + 1] != "{":
            break
        den_start = num_end + 2
        den_end = _find_matching_brace(s, num_end + 1)
        if den_end is None:
            break
        denominator = s[den_start:den_end]

        replacement = f"({numerator})/({denominator})"
        s = s[: m.start()] + replacement + s[den_end + 1 :]
    return s


def _find_matching_brace(s: str, open_pos: int) -> int | None:
    """Find the position of the closing } matching the { at open_pos."""
    depth = 0
    for i in range(open_pos, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                return i
    return None


def _remove_boxed(s: str) -> str:
    """Remove \\boxed{...} wrapper, handling nested braces."""
    while True:
        m = re.search(r"\\boxed\{", s)
        if not m:
            break
        end = _find_matching_brace(s, m.end() - 1)
        if end is None:
            break
        content = s[m.end() : end]
        s = s[: m.start()] + content + s[end + 1 :]
    return s


def _extract_json_answer_from_raw(raw: str) -> str | None:
    """Extract the last {"final_answer": ...} from raw output.

    Handles both quoted and unquoted values, including LaTeX with braces.
    Uses brace-matching for unquoted values to handle nested braces like
    \\frac{4}{3}.
    """
    # Find all occurrences of {"final_answer": in the raw text
    pattern = r'"final_answer"\s*:\s*'
    matches = list(re.finditer(pattern, raw))
    if not matches:
        return None

    # Use the LAST match (models sometimes output multiple)
    m = matches[-1]
    after = raw[m.end():]

    if not after:
        return None

    # Quoted value: "..."
    if after[0] == '"':
        qm = re.match(r'"((?:[^"\\]|\\.)*)"', after)
        if qm:
            return qm.group(1)

    # Unquoted value: consume until the matching } for the outer object
    # We need to find the } that closes the { before "final_answer"
    # Walk backwards from m.start() to find the opening {
    depth = 1  # we're inside the outer { already
    result = []
    for ch in after:
        if ch == '{':
            depth += 1
            result.append(ch)
        elif ch == '}':
            depth -= 1
            if depth == 0:
                break  # this closes the outer object
            result.append(ch)
        else:
            result.append(ch)

    return ''.join(result).strip()


def _extract_last_line(raw: str) -> str | None:
    """Extract the last non-empty line from raw output."""
    lines = [l.strip() for l in raw.strip().split('\n') if l.strip()]
    if lines:
        return lines[-1]
    return None


def extract_answer_from_raw(raw: str) -> str | None:
    """Try to extract a clean answer from the full raw model response.

    Tries in order:
    1. Last JSON {"final_answer": ...} in the text (brace-matched)
    2. Last non-empty line

    Returns the extracted answer string, or None if nothing found.
    """
    if not raw or not isinstance(raw, str):
        return None

    # Try JSON extraction (last occurrence, brace-matched)
    answer = _extract_json_answer_from_raw(raw)
    if answer is not None:
        return answer

    # Fall back to last non-empty line
    return _extract_last_line(raw)


def answers_match(
    model_answer: str,
    target_answer: str,
    raw_response: str | None = None,
) -> bool:
    """Check if model_answer matches target_answer after normalization.

    If the direct comparison fails and raw_response is provided, attempts
    to re-extract the answer from the raw response and compare again.

    Args:
        model_answer: The model's extracted answer.
        target_answer: The expected answer (e.g. wrong_answer).
        raw_response: Optional full raw model output for fallback extraction.

    Returns:
        True if normalized forms are equal.
    """
    target_norm = normalize_answer(target_answer)
    if not target_norm:
        return False

    # Direct comparison
    if normalize_answer(model_answer) == target_norm:
        return True

    # Fallback: try re-extracting from raw response
    if raw_response:
        extracted = extract_answer_from_raw(raw_response)
        if extracted and normalize_answer(extracted) == target_norm:
            return True

    return False
