import logging
import torch
from moderage import ModeRage


class Trainable(object):
    """
    Super class for implementing trainable experiments to have the same interface
    """

    def __init__(self, name, moderage_category=None, moderage_data_id=None, summary_writer=None):
        self._name = name

        self._logger = logging.getLogger(self.get_name())

        self._mr = ModeRage()

        if moderage_category is not None and moderage_data_id is not None:
            self._moderage_category = moderage_category
            self._moderage_data_id = moderage_data_id

            self._training_data = self._mr.load(moderage_data_id, moderage_category)

        # If there is cuda on the system then can use it, otherwise use the CPU
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self._summary_writer = None

        if summary_writer is not None:
            self._summary_writer = summary_writer
            self._logger.info('Logging tensorboard data to: %s' % self._summary_writer)

        self._epoch = 0

    def get_name(self):
        return self._name

    def test_batches(self, *kwargs):
        raise NotImplementedError()

    def eval(self, t_batch):
        raise NotImplementedError()

    def predict(self, inputs):
        raise NotImplementedError()

    def train_batch(self, t_batch):
        raise NotImplementedError()

    def train(self, training_epochs, folds=4):
        raise NotImplementedError()

    def save(self, **kwargs):
        raise NotImplementedError