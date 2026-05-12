import functools
import types
from ..utils import partition_tree_shuffle, MaskedModel
from .._explanation import Explanation
from ._explainer import Explainer
import numpy as np
import warnings
from .. import links
from .. import maskers
from ..maskers import Masker
from ..models import Model


class Permutation(Explainer):
    """ This method approximates the Shapley values by iterating through permutations of the inputs.

    This is a model agnostic explainer that guarantees local accuracy (additivity) by iterating completely
    through an entire permutation of the features in both forward and reverse directions (antithetic sampling).
    """

    def __init__(self, model, masker, link=links.identity, feature_names=None, linearize_link=True, seed=None, **call_args):
        """ Build an explainers.Permutation object for the given model using the given masker object.

        Parameters
        ----------
        model : function
            A callable python object that executes the model given a set of input data samples.

        masker : function or numpy.array or pandas.DataFrame
            A callable python object used to "mask" out hidden features of the form `masker(binary_mask, x)`.

        seed: None or int
            Seed for reproducibility
        """

        # setting seed for random generation
        np.random.seed(seed)

        super().__init__(model, masker, link=link, linearize_link=linearize_link, feature_names=feature_names)

        if not isinstance(self.model, Model):
            self.model = Model(self.model)

        # if we have gotten default arguments for the call function we need to wrap ourselves in a new class
        if len(call_args) > 0:
            class Permutation(self.__class__):
                def __call__(self, *args, max_evals=500, main_effects=False, error_bounds=False, batch_size="auto",
                             outputs=None, silent=False):
                    return super().__call__(
                        *args, max_evals=max_evals, main_effects=main_effects, error_bounds=error_bounds,
                        batch_size=batch_size, outputs=outputs, silent=silent
                    )
            Permutation.__call__.__doc__ = self.__class__.__call__.__doc__
            self.__class__ = Permutation
            for k, v in call_args.items():
                self.__call__.__kwdefaults__[k] = v

    def __call__(self, *args, max_evals=500, main_effects=False, error_bounds=False, batch_size="auto",
                 outputs=None, silent=False):
        """ Explain the output of the model on the given arguments.
        """
        return super().__call__(
            *args, max_evals=max_evals, main_effects=main_effects, error_bounds=error_bounds, batch_size=batch_size,
            outputs=outputs, silent=silent
        )

    def explain_row(self, *row_args, max_evals, main_effects, error_bounds, batch_size, outputs, silent):
        """ Explains a single row and returns the tuple (row_values, row_expected_values, row_mask_shapes).
        """

        # build a masked version of the model for the current input sample
        fm = MaskedModel(self.model, self.masker, self.link, self.linearize_link, *row_args)

        # by default we run 10 permutations forward and backward
        if max_evals == "auto":
            max_evals = 10 * 2 * len(fm)

        # compute any custom clustering for this row
        row_clustering = None
        if getattr(self.masker, "clustering", None) is not None:
            if isinstance(self.masker.clustering, np.ndarray):
                row_clustering = self.masker.clustering
            elif callable(self.masker.clustering):
                row_clustering = self.masker.clustering(*row_args)
            else:
                raise NotImplementedError("The masker passed has a .clustering attribute that is not yet supported by the Permutation explainer!")

        # loop over many permutations
        inds = fm.varying_inputs()
        inds_mask = np.zeros(len(fm), dtype=bool)
        inds_mask[inds] = True
        masks = np.zeros(2*len(inds)+1, dtype=int)
        masks[0] = MaskedModel.delta_mask_noop_value
        npermutations = max_evals // (2*len(inds)+1)
        row_values = None
        row_values_history = None
        history_pos = 0
        main_effect_values = None
        if len(inds) > 0:
            for _ in range(npermutations):

                # shuffle the indexes so we get a random permutation ordering
                if row_clustering is not None:
                    partition_tree_shuffle(inds, inds_mask, row_clustering)
                else:
                    np.random.shuffle(inds)

                # create a large batch of masks to evaluate
                i = 1
                for ind in inds:
                    masks[i] = ind
                    i += 1
                for ind in inds:
                    masks[i] = ind
                    i += 1

                # evaluate the masked model
                outputs = fm(masks, zero_index=0, batch_size=batch_size)

                if row_values is None:
                    row_values = np.zeros((len(fm),) + outputs.shape[1:])

                    if error_bounds:
                        row_values_history = np.zeros((2 * npermutations, len(fm),) + outputs.shape[1:])

                # update our SHAP value estimates
                i = 0
                for ind in inds:  # forward
                    row_values[ind] += outputs[i + 1] - outputs[i]
                    if error_bounds:
                        row_values_history[history_pos][ind] = outputs[i + 1] - outputs[i]
                    i += 1
                history_pos += 1
                for ind in inds:  # backward
                    row_values[ind] += outputs[i] - outputs[i + 1]
                    if error_bounds:
                        row_values_history[history_pos][ind] = outputs[i] - outputs[i + 1]
                    i += 1
                history_pos += 1

            if npermutations == 0:
                raise ValueError(f"max_evals={max_evals} is too low for the Permutation explainer, it must be at least 2 * num_features + 1 = {2 * len(inds) + 1}!")

            expected_value = outputs[0]

            # compute the main effects if we need to
            if main_effects:
                main_effect_values = fm.main_effects(inds, batch_size=batch_size)
        else:
            masks = np.zeros(1, dtype=int)
            outputs = fm(masks, zero_index=0, batch_size=1)
            expected_value = outputs[0]
            row_values = np.zeros((len(fm),) + outputs.shape[1:])
            if error_bounds:
                row_values_history = np.zeros((2 * npermutations, len(fm),) + outputs.shape[1:])

        return {
            "values": row_values / (2 * npermutations),
            "expected_values": expected_value,
            "mask_shapes": fm.mask_shapes,
            "main_effects": main_effect_values,
            "clustering": row_clustering,
            "error_std": None if row_values_history is None else row_values_history.std(0),
            "output_names": self.model.output_names if hasattr(self.model, "output_names") else None
        }

    def __str__(self):
        return "shap.explainers.Permutation()"
