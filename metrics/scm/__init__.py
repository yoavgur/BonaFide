"""SCM-based faithfulness metric.

Reference: Bao et al., "How Likely Do LLMs with CoT Mimic Human Reasoning?"
(COLING 2025).
"""

from metrics.scm.config import SCMConfig
from metrics.scm.metric import SCMMetric

__all__ = ["SCMMetric", "SCMConfig"]
