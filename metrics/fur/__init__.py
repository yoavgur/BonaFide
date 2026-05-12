"""FUR: Faithfulness by Unlearning Reasoning steps.

Measures parametric faithfulness by unlearning individual CoT steps from model
weights (NPO+KL) and checking if the model's answer changes.

Reference:
    Tutek et al., "Measuring Chain of Thought Faithfulness by Unlearning
    Reasoning Steps", EMNLP 2025.
    https://github.com/technion-cs-nlp/parametric-faithfulness
"""

from metrics.fur.metric import FURMetric

__all__ = ["FURMetric"]
