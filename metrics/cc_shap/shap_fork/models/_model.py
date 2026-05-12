import numpy as np
from ..utils import record_import_error, safe_isinstance
from .._serializable import Serializable

try:
    import torch
except ImportError as e:
    record_import_error("torch", "torch could not be imported!", e)


class Model(Serializable):
    """ This is the superclass of all models.
    """

    def __init__(self, model=None):
        """ Wrap a callable model as a SHAP Model object.
        """
        if isinstance(model, Model):
            self.inner_model = model.inner_model
        else:
            self.inner_model = model

        if hasattr(model, "output_names"):
            self.output_names = model.output_names

    def __call__(self, *args):
        out = self.inner_model(*args)
        is_tensor = safe_isinstance(out, "torch.Tensor")
        out = out.cpu().detach().numpy() if is_tensor else np.array(out)
        return out
