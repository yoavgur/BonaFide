"""Lightweight sentence splitting utilities.

This module is intentionally free of heavy dependencies (torch,
transformers, sentence_transformers, etc.) so that it can be imported
from contexts where CUDA must not be initialized — e.g. before vLLM
forks its engine subprocess.

The canonical implementation lives here; ``isolate_steps.py`` re-exports
``split_into_sentences`` for backward compatibility.
"""

from __future__ import annotations

from typing import List

_CLOSERS = set(['"', "\u201d", ")", "]", "}", "\u00bb"])


def split_into_sentences(text: str) -> List[str]:
    """
    Heuristic sentence splitter with special handling for *double-quote* quotes, ellipses, and NEWLINES as sentence breaks.

    Rules implemented:
    - Only double quotes (", \u201c, \u201d, \u00ab, \u00bb) participate in "inside quote" logic. Single quotes are ignored.
    - Don't split on '.', '!' or '?' if it's inside double quotes,
      unless it's at the end of the quoted span AND the next letter outside the quote is capitalized.
    - Don't split on "..." unless the next letter after it is capitalized (outside quotes) or end-of-text.
    - Every literal newline (\\n) is ALWAYS a sentence break.
    """
    if not text:
        return []

    sentences: List[str] = []
    buf: List[str] = []

    # Track whether we're inside a double-quoted span
    in_dquote = False

    def is_open_dquote(ch: str) -> bool:
        return ch in ['"', "\u201c", "\u00ab"]

    def is_close_dquote(ch: str) -> bool:
        return ch in ['"', "\u201d", "\u00bb"]

    def toggle_dquote(i: int) -> None:
        nonlocal in_dquote
        ch = text[i]
        prev_ch = text[i - 1] if i - 1 >= 0 else None
        if prev_ch == "\\":  # ignore escaped quotes like \"
            return
        if ch == '"' or ch in ["\u201c", "\u201d", "\u00ab", "\u00bb"]:
            # For ASCII " we toggle; for curly quotes we treat \u201c/\u00ab as open and \u201d/\u00bb as close.
            if ch == '"':
                in_dquote = not in_dquote
            elif ch in ["\u201c", "\u00ab"]:
                in_dquote = True
            elif ch in ["\u201d", "\u00bb"]:
                in_dquote = False

    def next_nonspace_index(start: int) -> int:
        j = start
        n = len(text)
        while j < n and text[j].isspace():
            j += 1
        return j

    def next_letter_is_capitalized_outside_quotes(start: int) -> bool:
        j = next_nonspace_index(start)
        n = len(text)

        # Skip closers like closing quotes/brackets after punctuation.
        while j < n and text[j] in _CLOSERS:
            j += 1
            j = next_nonspace_index(j)

        if j >= n:
            return True

        ch = text[j]
        return ch.isalpha() and ch.isupper()

    def consume_trailing_closers(i: int) -> int:
        """
        Include immediate closing quotes/brackets in the current sentence buffer.
        Returns the last consumed index so the caller can advance `i`.
        """
        n = len(text)
        j = i + 1
        while j < n and text[j] in _CLOSERS:
            buf.append(text[j])
            toggle_dquote(j)  # keep in_dquote consistent
            j += 1
        return j - 1

    i = 0
    n = len(text)

    while i < n:
        ch = text[i]

        # Newline is ALWAYS a sentence break.
        if ch == "\n":
            s = "".join(buf).strip()
            if s:
                sentences.append(s)
            buf = []
            i += 1
            continue

        buf.append(ch)
        toggle_dquote(i)

        # Ellipsis "..."
        if ch == "." and i + 2 < n and text[i : i + 3] == "...":
            buf.append(".")
            buf.append(".")
            i += 2  # now at last '.'

            if next_letter_is_capitalized_outside_quotes(i + 1):
                i = consume_trailing_closers(i)
                s = "".join(buf).strip()
                if s:
                    sentences.append(s)
                buf = []

            i += 1
            continue

        # End punctuation
        if ch in [".", "!", "?"]:
            if in_dquote:
                # Only split if we're at the end of the quoted span (i.e., followed by a closing quote/bracket)
                # AND the next letter outside is capitalized.
                j = next_nonspace_index(i + 1)
                if j < n and text[j] in _CLOSERS:
                    if next_letter_is_capitalized_outside_quotes(i + 1):
                        i = consume_trailing_closers(i)
                        s = "".join(buf).strip()
                        if s:
                            sentences.append(s)
                        buf = []
            else:
                if next_letter_is_capitalized_outside_quotes(i + 1):
                    i = consume_trailing_closers(i)
                    s = "".join(buf).strip()
                    if s:
                        sentences.append(s)
                    buf = []

        i += 1

    tail = "".join(buf).strip()
    if tail:
        sentences.append(tail)

    return sentences
