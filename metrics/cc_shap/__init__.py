"""CC-SHAP: Measuring self-consistency of chain-of-thought explanations.

Computes SHAP-based input contribution similarity between answer prediction
and explanation generation. High similarity indicates the explanation is
consistent with the model's decision process.

Reference:
    Parcalabescu & Frank (2024). "On Measuring Faithfulness or Self-consistency
    of Natural Language Explanations." ACL 2024.
    https://github.com/Heidelberg-NLP/CC-SHAP
"""

from metrics.cc_shap.config import CCSHAPConfig
from metrics.cc_shap.metric import CCSHAPMetric

__all__ = ["CCSHAPMetric", "CCSHAPConfig"]
