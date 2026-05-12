import copy
import time
import numpy as np
import scipy as sp
from .. import maskers
from .. import links
from ..utils import safe_isinstance, show_progress
from ..utils.transformers import is_transformers_lm
from .. import models
from ..models import Model
from ..maskers import Masker
from .._explanation import Explanation
from .._serializable import Serializable
from .. import explainers
from ..utils._exceptions import InvalidAlgorithmError


class Explainer(Serializable):
    """ Uses Shapley values to explain any machine learning model or python function.

    This is the primary explainer interface for the SHAP library. It takes any combination
    of a model and masker and returns a callable subclass object that implements
    the particular estimation algorithm that was chosen.
    """

    def __init__(self, model, masker=None, link=links.identity, algorithm="auto", output_names=None, feature_names=None, linearize_link=True,
                 seed=None, **kwargs):
        """ Build a new explainer for the passed model.

        Parameters
        ----------
        model : object or function
            User supplied function or model object that takes a dataset of samples and
            computes the output of the model for those samples.

        masker : function, numpy.array, pandas.DataFrame, tokenizer, None, or a list of these for each model input
            The function used to "mask" out hidden features of the form `masked_args = masker(*model_args, mask=mask)`.

        link : function
            The link function used to map between the output units of the model and the SHAP value units.

        algorithm : "auto", "permutation", "partition", or "exact"
            The algorithm used to estimate the Shapley values.

        output_names : None or list of strings
            The names of the model outputs.

        seed: None or int
            seed for reproducibility
        """

        self.model = model
        self.output_names = output_names
        self.feature_names = feature_names

        # wrap the incoming masker object as a Masker object
        if safe_isinstance(masker, "pandas.core.frame.DataFrame") or \
                ((safe_isinstance(masker, "numpy.ndarray") or sp.sparse.issparse(masker)) and len(masker.shape) == 2):
            if algorithm == "partition":
                self.masker = maskers.Partition(masker)
            else:
                self.masker = maskers.Independent(masker)
        elif safe_isinstance(masker, ["transformers.PreTrainedTokenizer", "transformers.tokenization_utils_base.PreTrainedTokenizerBase"]):
            if is_transformers_lm(self.model):
                # auto assign text infilling if model is a transformer model with lm head
                self.masker = maskers.Text(masker, mask_token="...", collapse_mask_token=True)
            else:
                self.masker = maskers.Text(masker)
        elif (masker is list or masker is tuple) and masker[0] is not str:
            self.masker = maskers.Composite(*masker)
        elif (masker is dict) and ("mean" in masker):
            self.masker = maskers.Independent(masker)
        else:
            self.masker = masker

        # Check for transformer pipeline objects and wrap them
        if safe_isinstance(self.model, "transformers.pipelines.Pipeline"):
            if is_transformers_lm(self.model.model):
                return self.__init__(  # pylint: disable=non-parent-init-called
                    self.model.model, self.model.tokenizer if self.masker is None else self.masker,
                    link=link, algorithm=algorithm, output_names=output_names, feature_names=feature_names, linearize_link=linearize_link, **kwargs
                )

        # wrap self.masker and self.model for output text explanation algorithm
        if is_transformers_lm(self.model):
            self.model = models.TeacherForcing(self.model, self.masker.tokenizer)
            self.masker = maskers.OutputComposite(self.masker, self.model.text_generate)
        elif safe_isinstance(self.model, "shap.models.TeacherForcing") and safe_isinstance(self.masker, ["shap.maskers.Text", "shap.maskers.Image"]):
            self.masker = maskers.OutputComposite(self.masker, self.model.text_generate)

        # validate and save the link function
        if callable(link):
            self.link = link
        else:
            raise TypeError("The passed link function needs to be callable!")
        self.linearize_link = linearize_link

        # if we are called directly (as opposed to through super()) then we convert ourselves to the subclass
        # that implements the specific algorithm that was chosen
        if self.__class__ is Explainer:

            # do automatic algorithm selection
            if algorithm == "auto":

                # use model agnostic methods
                if callable(self.model):
                    if issubclass(type(self.masker), maskers.Independent):
                        algorithm = "permutation"
                    elif issubclass(type(self.masker), maskers.Partition):
                        algorithm = "permutation"
                    elif (getattr(self.masker, "text_data", False) or getattr(self.masker, "image_data", False)) and hasattr(self.masker, "clustering"):
                        algorithm = "partition"
                    else:
                        algorithm = "permutation"

                # if we get here then we don't know how to handle what was given to us
                else:
                    raise TypeError("The passed model is not callable and cannot be analyzed directly with the given masker! Model: " + str(model))

            # build the right subclass
            if algorithm == "permutation":
                self.__class__ = explainers.Permutation
                explainers.Permutation.__init__(self, self.model, self.masker, link=self.link, feature_names=self.feature_names, linearize_link=linearize_link, seed=seed, **kwargs)
            elif algorithm == "partition":
                self.__class__ = explainers.Partition
                explainers.Partition.__init__(self, self.model, self.masker, link=self.link, feature_names=self.feature_names, linearize_link=linearize_link, output_names=self.output_names, **kwargs)
            else:
                raise InvalidAlgorithmError("Unknown algorithm type passed: %s!" % algorithm)


    def __call__(self, *args, max_evals="auto", main_effects=False, error_bounds=False, batch_size="auto",
                 outputs=None, silent=False, **kwargs):
        """ Explains the output of model(*args), where args is a list of parallel iteratable datasets.
        """

        start_time = time.time()

        if issubclass(type(self.masker), maskers.OutputComposite) and len(args)==2:
            self.masker.model = models.TextGeneration(target_sentences=args[1])
            args = args[:1]
        # parse our incoming arguments
        num_rows = None
        args = list(args)
        if self.feature_names is None:
            feature_names = [None for _ in range(len(args))]
        elif issubclass(type(self.feature_names[0]), (list, tuple)):
            feature_names = copy.deepcopy(self.feature_names)
        else:
            feature_names = [copy.deepcopy(self.feature_names)]
        for i in range(len(args)):

            # try and see if we can get a length from any of the for our progress bar
            if num_rows is None:
                try:
                    num_rows = len(args[i])
                except Exception:  # pylint: disable=broad-except
                    pass

            # convert DataFrames to numpy arrays
            if safe_isinstance(args[i], "pandas.core.frame.DataFrame"):
                feature_names[i] = list(args[i].columns)
                args[i] = args[i].to_numpy()

            # convert nlp Dataset objects to lists
            if safe_isinstance(args[i], "nlp.arrow_dataset.Dataset"):
                args[i] = args[i]["text"]
            elif issubclass(type(args[i]), dict) and "text" in args[i]:
                args[i] = args[i]["text"]

        if batch_size == "auto":
            if hasattr(self.masker, "default_batch_size"):
                batch_size = self.masker.default_batch_size
            else:
                batch_size = 10

        # loop over each sample, filling in the values array
        values = []
        output_indices = []
        expected_values = []
        mask_shapes = []
        main_effects = []
        hierarchical_values = []
        clustering = []
        output_names = []
        error_std = []
        if callable(getattr(self.masker, "feature_names", None)):
            feature_names = [[] for _ in range(len(args))]
        for row_args in show_progress(zip(*args), num_rows, self.__class__.__name__+" explainer", silent):
            row_result = self.explain_row(
                *row_args, max_evals=max_evals, main_effects=main_effects, error_bounds=error_bounds,
                batch_size=batch_size, outputs=outputs, silent=silent, **kwargs
            )
            values.append(row_result.get("values", None))
            output_indices.append(row_result.get("output_indices", None))
            expected_values.append(row_result.get("expected_values", None))
            mask_shapes.append(row_result["mask_shapes"])
            main_effects.append(row_result.get("main_effects", None))
            clustering.append(row_result.get("clustering", None))
            hierarchical_values.append(row_result.get("hierarchical_values", None))
            tmp = row_result.get("output_names", None)
            output_names.append(tmp(*row_args) if callable(tmp) else tmp)
            error_std.append(row_result.get("error_std", None))
            if callable(getattr(self.masker, "feature_names", None)):
                row_feature_names = self.masker.feature_names(*row_args)
                for i in range(len(row_args)):
                    feature_names[i].append(row_feature_names[i])

        # split the values up according to each input
        arg_values = [[] for a in args]
        for i, v in enumerate(values):
            pos = 0
            for j in range(len(args)):
                mask_length = np.prod(mask_shapes[i][j])
                arg_values[j].append(values[i][pos:pos+mask_length])
                pos += mask_length

        # collapse the arrays as possible
        expected_values = pack_values(expected_values)
        main_effects = pack_values(main_effects)
        output_indices = pack_values(output_indices)
        main_effects = pack_values(main_effects)
        hierarchical_values = pack_values(hierarchical_values)
        error_std = pack_values(error_std)
        clustering = pack_values(clustering)

        # getting output labels
        ragged_outputs = False
        if output_indices is not None:
            ragged_outputs = not all(len(x) == len(output_indices[0]) for x in output_indices)
        if self.output_names is None:
            if None not in output_names:
                if not ragged_outputs:
                    sliced_labels = np.array(output_names)
                else:
                    sliced_labels = [np.array(output_names[i])[index_list] for i,index_list in enumerate(output_indices)]
            else:
                sliced_labels = None
        else:
            assert output_indices is not None, "You have passed a list for output_names but the model seems to not have multiple outputs!"
            labels = np.array(self.output_names)
            sliced_labels = [labels[index_list] for index_list in output_indices]
            if not ragged_outputs:
                sliced_labels = np.array(sliced_labels)

        if isinstance(sliced_labels, np.ndarray) and len(sliced_labels.shape) == 2:
            if np.all(sliced_labels[0,:] == sliced_labels):
                sliced_labels = sliced_labels[0]

        # allow the masker to transform the input data to better match the masking pattern
        if hasattr(self.masker, "data_transform"):
            new_args = []
            for row_args in zip(*args):
                new_args.append([pack_values(v) for v in self.masker.data_transform(*row_args)])
            args = list(zip(*new_args))

        # build the explanation objects
        out = []
        for j, data in enumerate(args):

            # reshape the attribution values using the mask_shapes
            tmp = []
            for i, v in enumerate(arg_values[j]):
                if np.prod(mask_shapes[i][j]) != np.prod(v.shape):  # see if we have multiple outputs
                    tmp.append(v.reshape(*mask_shapes[i][j], -1))
                else:
                    tmp.append(v.reshape(*mask_shapes[i][j]))
            arg_values[j] = pack_values(tmp)

            if feature_names[j] is None:
                feature_names[j] = ["Feature " + str(i) for i in range(data.shape[1])]

            # build an explanation object for this input argument
            out.append(Explanation(
                arg_values[j], expected_values, data,
                feature_names=feature_names[j], main_effects=main_effects,
                clustering=clustering,
                hierarchical_values=hierarchical_values,
                output_names=sliced_labels,
                error_std=error_std,
                compute_time=time.time() - start_time
            ))
        return out[0] if len(out) == 1 else out

    def explain_row(self, *row_args, max_evals, main_effects, error_bounds, outputs, silent, **kwargs):
        """ Explains a single row and returns the tuple (row_values, row_expected_values, row_mask_shapes).

        This is an abstract method meant to be implemented by each subclass.
        """
        return {}

    @staticmethod
    def supports_model_with_masker(model, masker):
        """ Determines if this explainer can handle the given model.

        This is an abstract static method meant to be implemented by each subclass.
        """
        return False

    @staticmethod
    def _compute_main_effects(fm, expected_value, inds):
        """ A utility method to compute the main effects from a MaskedModel.
        """

        # mask each input on in isolation
        masks = np.zeros(2*len(inds)-1, dtype=int)
        last_ind = -1
        for i in range(len(inds)):
            if i > 0:
                masks[2*i - 1] = -last_ind - 1  # turn off the last input
            masks[2*i] = inds[i]  # turn on this input
            last_ind = inds[i]

        # compute the main effects for the given indexes
        main_effects = fm(masks) - expected_value

        # expand the vector to the full input size
        expanded_main_effects = np.zeros(len(fm))
        for i, ind in enumerate(inds):
            expanded_main_effects[ind] = main_effects[i]

        return expanded_main_effects


def pack_values(values):
    """ Used the clean up arrays before putting them into an Explanation object.
    """

    if not hasattr(values, "__len__"):
        return values

    # collapse the values if we didn't compute them
    if values is None or values[0] is None:
        return None

    # convert to a single numpy matrix when the array is not ragged
    elif np.issubdtype(type(values[0]), np.number) or len(np.unique([len(v) for v in values])) == 1:
        return np.array(values)
    else:
        return np.array(values, dtype=object)
