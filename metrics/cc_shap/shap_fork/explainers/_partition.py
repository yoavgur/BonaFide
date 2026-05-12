import types
import copy
import inspect
from ..utils import MaskedModel
import numpy as np
import warnings
import time
from tqdm.auto import tqdm
import queue
from ..utils import assert_import, record_import_error, safe_isinstance, make_masks, OpChain
from .. import Explanation
from .. import maskers
from ._explainer import Explainer
from .. import links
from ..maskers import Masker
from ..models import Model

# .shape[0] messes up pylint a lot here
# pylint: disable=unsubscriptable-object


class Partition(Explainer):

    def __init__(self, model, masker, *, output_names=None, link=links.identity, linearize_link=True,
                 feature_names=None, **call_args):
        """ Uses the Partition SHAP method to explain the output of any function.

        Partition SHAP computes Shapley values recursively through a hierarchy of features, this
        hierarchy defines feature coalitions and results in the Owen values from game theory.
        """

        super().__init__(model, masker, link=link, linearize_link=linearize_link, algorithm="partition",
                         output_names=output_names, feature_names=feature_names)

        self.input_shape = masker.shape[1:] if hasattr(masker, "shape") and not callable(masker.shape) else None
        if not safe_isinstance(self.model, "shap.models.Model"):
            self.model = Model(self.model)
        self.expected_value = None
        self._curr_base_value = None
        if getattr(self.masker, "clustering", None) is None:
            raise ValueError("The passed masker must have a .clustering attribute defined! Try shap.maskers.Partition(data) for example.")

        # handle higher dimensional tensor inputs
        if self.input_shape is not None and len(self.input_shape) > 1:
            self._reshaped_model = lambda x: self.model(x.reshape(x.shape[0], *self.input_shape))
        else:
            self._reshaped_model = self.model

        # if we don't have a dynamic clustering algorithm then can precompute
        # a lot of information
        if not callable(self.masker.clustering):
            self._clustering = self.masker.clustering
            self._mask_matrix = make_masks(self._clustering)

        # if we have gotten default arguments for the call function we need to wrap ourselves in a new class that
        # has a call function with those new default arguments
        if len(call_args) > 0:
            class Partition(self.__class__):
                # this signature should match the __call__ signature of the class defined below
                def __call__(self, *args, max_evals=500, fixed_context=None, main_effects=False, error_bounds=False, batch_size="auto",
                             outputs=None, silent=False):
                    return super().__call__(
                        *args, max_evals=max_evals, fixed_context=fixed_context, main_effects=main_effects, error_bounds=error_bounds,
                        batch_size=batch_size, outputs=outputs, silent=silent
                    )
            Partition.__call__.__doc__ = self.__class__.__call__.__doc__
            self.__class__ = Partition
            for k, v in call_args.items():
                self.__call__.__kwdefaults__[k] = v

    # note that changes to this function signature should be copied to the default call argument wrapper above
    def __call__(self, *args, max_evals=500, fixed_context=None, main_effects=False, error_bounds=False, batch_size="auto",
                 outputs=None, silent=False):
        """ Explain the output of the model on the given arguments.
        """
        return super().__call__(
            *args, max_evals=max_evals, fixed_context=fixed_context, main_effects=main_effects, error_bounds=error_bounds, batch_size=batch_size,
            outputs=outputs, silent=silent
        )

    def explain_row(self, *row_args, max_evals, main_effects, error_bounds, batch_size, outputs, silent, fixed_context="auto"):
        """ Explains a single row and returns the tuple (row_values, row_expected_values, row_mask_shapes).
        """

        if fixed_context == "auto":
            fixed_context = None
        elif fixed_context not in [0, 1, None]:
            raise ValueError("Unknown fixed_context value passed (must be 0, 1 or None): %s" % fixed_context)

        # build a masked version of the model for the current input sample
        fm = MaskedModel(self.model, self.masker, self.link, self.linearize_link, *row_args)

        # make sure we have the base value and current value outputs
        M = len(fm)
        m00 = np.zeros(M, dtype=bool)
        # if not fixed background or no base value assigned then compute base value for a row
        if self._curr_base_value is None or not getattr(self.masker, "fixed_background", False):
            self._curr_base_value = fm(m00.reshape(1, -1), zero_index=0)[0]  # the zero index param tells the masked model what the baseline is
        f11 = fm(~m00.reshape(1, -1))[0]

        if callable(self.masker.clustering):
            self._clustering = self.masker.clustering(*row_args)
            self._mask_matrix = make_masks(self._clustering)

        if hasattr(self._curr_base_value, 'shape') and len(self._curr_base_value.shape) > 0:
            if outputs is None:
                outputs = np.arange(len(self._curr_base_value))
            elif isinstance(outputs, OpChain):
                outputs = outputs.apply(Explanation(f11)).values

            out_shape = (2*self._clustering.shape[0]+1, len(outputs))
        else:
            out_shape = (2*self._clustering.shape[0]+1,)

        if max_evals == "auto":
            max_evals = 500

        self.values = np.zeros(out_shape)
        self.dvalues = np.zeros(out_shape)

        self.owen(fm, self._curr_base_value, f11, max_evals - 2, outputs, fixed_context, batch_size, silent)

        # drop the interaction terms down onto self.values
        self.values[:] = self.dvalues

        lower_credit(len(self.dvalues) - 1, 0, M, self.values, self._clustering)

        return {
            "values": self.values[:M].copy(),
            "expected_values": self._curr_base_value if outputs is None else self._curr_base_value[outputs],
            "mask_shapes": [s + out_shape[1:] for s in fm.mask_shapes],
            "main_effects": None,
            "hierarchical_values": self.dvalues.copy(),
            "clustering": self._clustering,
            "output_indices": outputs,
            "output_names": getattr(self.model, "output_names", None)
        }

    def __str__(self):
        return "shap.explainers.Partition()"

    def owen(self, fm, f00, f11, max_evals, output_indexes, fixed_context, batch_size, silent):
        """ Compute a nested set of recursive Owen values based on an ordering recursion.
        """

        M = len(fm)
        m00 = np.zeros(M, dtype=bool)
        base_value = f00
        ind = len(self.dvalues)-1

        # make sure output_indexes is a list of indexes
        if output_indexes is not None:
            f00 = f00[output_indexes]
            f11 = f11[output_indexes]

        q = queue.PriorityQueue()
        q.put((0, 0, (m00, f00, f11, ind, 1.0)))
        eval_count = 0
        total_evals = min(max_evals, (M-1)*M)
        pbar = None
        start_time = time.time()
        while not q.empty():

            # if we passed our execution limit then leave everything else on the internal nodes
            if eval_count >= max_evals:
                while not q.empty():
                    m00, f00, f11, ind, weight = q.get()[2]
                    self.dvalues[ind] += (f11 - f00) * weight
                break

            # create a batch of work to do
            batch_args = []
            batch_masks = []
            while not q.empty() and len(batch_masks) < batch_size and eval_count + len(batch_masks) < max_evals:

                # get our next set of arguments
                m00, f00, f11, ind, weight = q.get()[2]

                # get the left and right children of this cluster
                lind = int(self._clustering[ind-M, 0]) if ind >= M else -1
                rind = int(self._clustering[ind-M, 1]) if ind >= M else -1

                # get the distance of this cluster's children
                if ind < M:
                    distance = -1
                else:
                    if self._clustering.shape[1] >= 3:
                        distance = self._clustering[ind-M, 2]
                    else:
                        distance = 1

                # check if we are a leaf node (or other negative distance cluster) and so should terminate our decent
                if distance < 0:
                    self.dvalues[ind] += (f11 - f00) * weight
                    continue

                # build the masks
                m10 = m00.copy()  # we separate the copy from the add so as to not get converted to a matrix
                m10[:] += self._mask_matrix[lind, :]
                m01 = m00.copy()
                m01[:] += self._mask_matrix[rind, :]

                batch_args.append((m00, m10, m01, f00, f11, ind, lind, rind, weight))
                batch_masks.append(m10)
                batch_masks.append(m01)

            batch_masks = np.array(batch_masks)

            # run the batch
            if len(batch_args) > 0:
                fout = fm(batch_masks)
                if output_indexes is not None:
                    fout = fout[:,output_indexes]

                eval_count += len(batch_masks)

                if pbar is None and time.time() - start_time > 5:
                    pbar = tqdm(total=total_evals, disable=silent, leave=False)
                    pbar.update(eval_count)
                if pbar is not None:
                    pbar.update(len(batch_masks))

            # use the results of the batch to add new nodes
            for i in range(len(batch_args)):

                m00, m10, m01, f00, f11, ind, lind, rind, weight = batch_args[i]

                # get the evaluated model output on the two new masked inputs
                f10 = fout[2*i]
                f01 = fout[2*i+1]

                new_weight = weight
                if fixed_context is None:
                    new_weight /= 2
                elif fixed_context == 0:
                    self.dvalues[ind] += (f11 - f10 - f01 + f00) * weight  # leave the interaction effect on the internal node
                elif fixed_context == 1:
                    self.dvalues[ind] -= (f11 - f10 - f01 + f00) * weight  # leave the interaction effect on the internal node

                if fixed_context is None or fixed_context == 0:
                    # recurse on the left node with zero context
                    args = (m00, f00, f10, lind, new_weight)
                    q.put((-np.max(np.abs(f10 - f00)) * new_weight, np.random.randn(), args))

                    # recurse on the right node with zero context
                    args = (m00, f00, f01, rind, new_weight)
                    q.put((-np.max(np.abs(f01 - f00)) * new_weight, np.random.randn(), args))

                if fixed_context is None or fixed_context == 1:
                    # recurse on the left node with one context
                    args = (m01, f01, f11, lind, new_weight)
                    q.put((-np.max(np.abs(f11 - f01)) * new_weight, np.random.randn(), args))

                    # recurse on the right node with one context
                    args = (m10, f10, f11, rind, new_weight)
                    q.put((-np.max(np.abs(f11 - f10)) * new_weight, np.random.randn(), args))

        if pbar is not None:
            pbar.close()

        self.last_eval_count = eval_count

        return output_indexes, base_value


def output_indexes_len(output_indexes):
    if output_indexes.startswith("max("):
        return int(output_indexes[4:-1])
    elif output_indexes.startswith("min("):
        return int(output_indexes[4:-1])
    elif output_indexes.startswith("max(abs("):
        return int(output_indexes[8:-2])
    elif not isinstance(output_indexes, str):
        return len(output_indexes)


def lower_credit(i, value, M, values, clustering):
    """Plain-Python replacement for the numba-jitted version."""
    if i < M:
        values[i] += value
        return
    li = int(clustering[i-M, 0])
    ri = int(clustering[i-M, 1])
    group_size = int(clustering[i-M, 3])
    lsize = int(clustering[li-M, 3]) if li >= M else 1
    rsize = int(clustering[ri-M, 3]) if ri >= M else 1
    assert lsize + rsize == group_size
    values[i] += value
    lower_credit(li, values[i] * lsize / group_size, M, values, clustering)
    lower_credit(ri, values[i] * rsize / group_size, M, values, clustering)
