from collections import defaultdict

import torch
import logging
from moderage import ModeRage
import numpy as np

from training.rolling_window import RollingWindow


class LossComponentCollector():

    def __init__(self, window_size=None):
        self._collect_windows = window_size is not None

        if self._collect_windows:
            self._loss_components_windows = defaultdict(lambda: RollingWindow(window_size))
        self._loss_components = {}

    def append_loss_components_batch(self, loss_components_batch):

        for k, loss_component in loss_components_batch.items():
            if self._collect_windows:
                self._loss_components_windows[k].add(loss_component)
            if k in self._loss_components:
                self._loss_components[k] = np.concatenate(
                    (self._loss_components[k], np.array(loss_component, dtype=np.float64).reshape(1)))
            else:
                self._loss_components[k] = np.array(loss_component, dtype=np.float64).reshape(1)

    def get_window_mean(self):
        return {k: v.mean() for k, v in self._loss_components_windows.items()}

    def get_window_std(self):
        return {k: v.std() for k, v in self._loss_components_windows.items()}

    def get_window_min(self):
        return {k: v.min() for k, v in self._loss_components_windows.items()}

    def get_window_max(self):
        return {k: v.max() for k, v in self._loss_components_windows.items()}

    def get_means(self):
        return {k: np.mean(v) for k, v in self._loss_components.items()}

    def get_history(self):
        return self._loss_components
