"""Faithfulness metrics framework.

Provides a uniform interface for measuring CoT faithfulness via different methods.
Each metric implements the FaithfulnessMetric ABC and receives a MetricContext.
"""

from metrics.adding_mistakes import AddingMistakesMetric
from metrics.base import FaithfulnessMetric, MetricContext
from metrics.cc_shap import CCSHAPMetric
from metrics.early_answering import EarlyAnsweringMetric
from metrics.filler_tokens import FillerTokensMetric
from metrics.fur import FURMetric
from metrics.paraphrasing import ParaphrasingMetric
from metrics.regex_baseline import RegexBaselineMetric
from metrics.scm import SCMMetric
from metrics.simulatability import SimulatabilityMetric

__all__ = [
    "FaithfulnessMetric",
    "MetricContext",
    "FURMetric",
    "CCSHAPMetric",
    "SCMMetric",
    "SimulatabilityMetric",
    "EarlyAnsweringMetric",
    "AddingMistakesMetric",
    "ParaphrasingMetric",
    "FillerTokensMetric",
    "RegexBaselineMetric",
]
