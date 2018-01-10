

class Evaluator(object):
    """Evaluator interface."""

    def dump(self, bluewhale, x, y,
             elementwise_score={},
             combined_score={},
             datatags=[],
             modeltags=[],
             batch_size=None,
             use_multiprocessing=False):
        """Dumps the result of an evaluation into a container.

        By default, the model will dump the evaluation metrics defined
        in keras.models.Model.compile.

        Parameters
        ----------
        bluewhale : :class:`BlueWhale`
            BlueWhale model to evaluate.
        x : :class:`BwDataset`
            Input dataset.
        y : :class:`BwDataset`
            Output dataset
        elementwise_score : dict
            Element-wise scores for multi-dimensional output data, which
            is applied to each output dimension separately. Default: \{\}.
        combined_score : dict
            Combined score for multi-dimensional output data applied across
            all dimensions toghether. For example, average AUC across all
            output dimensions. Default: \{\}.
        datatags : list
            List of dataset tags to be recorded. Default: [].
        modeltags : list
            List of modeltags to be recorded. Default: [].
        batch_size : int or None
            Batchsize used to enumerate the dataset. Default: None means a
            batch_size of 32 is used.
        use_multiprocessing : bool
            Use multiprocess threading for evaluating the results.
            Default: False.
        """
        pass
