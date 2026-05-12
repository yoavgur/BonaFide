from __future__ import annotations

import re
from dataclasses import dataclass, field


# Default stoplist used when extracting content words from a ground_truth_step.
_DEFAULT_STOPWORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "of", "to", "in", "on", "at", "for", "and", "or", "but", "if", "then",
    "else", "so", "as", "by", "with", "from", "that", "this", "these",
    "those", "it", "its", "we", "you", "i", "he", "she", "they", "them",
    "his", "her", "our", "their", "not", "no", "do", "does", "did",
    "have", "has", "had", "will", "would", "can", "could", "should",
    "may", "might", "must", "shall", "there", "here", "what", "which",
    "who", "whom", "how", "why", "when", "where",
})


@dataclass
class RegexBaselineConfig:
    # Pattern applied to the CoT for hint rows. Compiled with re.IGNORECASE
    # plus any extra flags from ``hint_pattern_flags``.
    hint_pattern: str = r"hint"
    hint_pattern_flags: int = 0  # extra flags OR'd onto re.IGNORECASE

    # Outright mode: ignore these tokens when collecting content words.
    outright_stopwords: frozenset = field(default_factory=lambda: _DEFAULT_STOPWORDS)
    outright_min_word_len: int = 3
