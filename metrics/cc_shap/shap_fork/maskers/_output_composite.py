from ._masker import Masker


class OutputComposite(Masker):
    """ A masker that is a combination of a masker and a model and outputs both masked args and the model's output.
    """

    def __init__(self, masker, model):
        """ Creates a masker from an underlying masker and model.

        This masker returns the masked input along with the model output for the passed args.

        Parameters
        ----------
        masker: object
            An object of the Masker base class (eg. Text/Image masker).

        model: object
            An object of Model base class used to generate output.

        Returns
        -------
        tuple
            A tuple consisting of the masked input using the underlying masker appended with the model output for passed args.
        """
        self.masker = masker
        self.model = model

        # copy attributes from the masker we are wrapping
        masker_attributes = ["shape", "invariants", "clustering", "data_transform", "mask_shapes", "feature_names", "text_data", "image_data"]
        for masker_attribute in masker_attributes:
            if getattr(self.masker, masker_attribute, None) is not None:
                setattr(self, masker_attribute, getattr(self.masker, masker_attribute))

    def __call__(self, mask, *args):
        """ Mask the args using the masker and return a tuple containing the masked input and the model output on the args.
        """
        masked_X = self.masker(mask, *args)
        y = self.model(*args)
        # wrap model output
        if not isinstance(y, tuple):
            y = (y,)
        # wrap masked input
        if not isinstance(masked_X, tuple):
            masked_X = (masked_X,)
        return masked_X + y
