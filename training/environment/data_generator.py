import logging
import numpy as np


class DataGenerator():

    def __init__(self, name):
        self._name = name

        self._logger = logging.getLogger(name)

    def get_name(self):
        return self._name

    def get_generator_params(self):
        raise NotImplementedError

    @staticmethod
    def convert_to_one_hot(x, range):
        oh = np.zeros((x.shape[0], range))
        oh[np.arange(x.shape[0]), x] = 1
        return oh

    def generate_samples(self, batch_size):
        """
        Batches must be generated in the form (batch, channels, width, height)
        :param num_batches:
        :param batch_size:
        :return:
        """
        raise NotImplementedError

    def get_data_shape(self):
        raise NotImplementedError