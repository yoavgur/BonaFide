"""Paraphrasing faithfulness metric.

Reference: Lanham et al., "Measuring Faithfulness in Chain-of-Thought
Reasoning" (arXiv 2307.13702), Section 2.6.
"""

from metrics.paraphrasing.config import ParaphrasingConfig
from metrics.paraphrasing.metric import ParaphrasingMetric, ParaphrasingResult

__all__ = ["ParaphrasingConfig", "ParaphrasingMetric", "ParaphrasingResult"]
