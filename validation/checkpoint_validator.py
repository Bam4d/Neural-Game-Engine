from collections import defaultdict

import numpy as np
import pandas as pd
import logging

from training.environment.gvgai_data_generator import GVGAILevelDataGenerator
from validation.prediction_accuracy import PredictionAccuracyMeasure


class CallBackValidator():

    def __init__(self, validation_repeats, validation_steps, ngpu_iterations, levels):
        self._logger = logging.getLogger("Generalisation")
        self._levels = levels
        self._level_shapes = {}
        self._history = {k: defaultdict(lambda: defaultdict(lambda: [])) for k in self._levels}

        self._repeats = validation_repeats
        self._steps = validation_steps
        self._ngpu_iterations = ngpu_iterations

        self._normal_level_generator = GVGAILevelDataGenerator(self._levels)

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        self._normal_level_generator.cleanup()

    def get_callback(self, model, summary_writer):

        num_actions = self._normal_level_generator.get_num_actions()
        action_mapping = self._normal_level_generator.get_action_mapping()
        compare_levels = [{'id': level, 'name': level} for level in self._levels]
        prediction_accuracy_measure = PredictionAccuracyMeasure(model._model, compare_levels,
                                                                ngpu_iterations=self._ngpu_iterations,
                                                                num_actions=num_actions,
                                                                action_mapping=action_mapping)

        def checkpoint_callback(e):
            # Then print a trace of learning a value
            for i, batch in enumerate(self._normal_level_generator.generate_samples(32)):
                t_batch = model._model.prepare_batch(batch)
                # Only view out the largest
                trace = i == len(self._levels) - 1
                level = self._levels[i]

                if level not in self._level_shapes:
                    self._level_shapes[level] = t_batch['input_observation_batch'].shape[2:]

                (loss, _), predicted = model.eval(t_batch, trace=trace)
                self._logger.info(f'Loss at {level}: {loss:.4f}')

            results = prediction_accuracy_measure.calculate(self._steps, self._repeats)

            for k, prediction_accuracy_error in results.items():

                for level, data in prediction_accuracy_error.items():
                    mean = np.mean(data)
                    max = np.max(data)
                    min = np.min(data)
                    std = np.std(data)

                    self._history[level][k]['epoch'].append(e)
                    self._history[level][k]['max'].append(max)
                    self._history[level][k]['mean'].append(mean)
                    self._history[level][k]['min'].append(min)
                    self._history[level][k]['std'].append(std)

                    if k == 'tile_error_data':
                        total_tiles = np.sum(self._level_shapes[level])
                        average_accuracy = (total_tiles - max) / total_tiles
                        summary_writer.add_scalars(f'NGE-Validation-{k}/{level}',
                                                   {'acc': average_accuracy}, e)

                        self._history[level][k]['acc'].append(average_accuracy)
                        self._logger.info(
                            f'Prediction Accuracy ({k}) for {self._steps} steps accuracy: {average_accuracy:.4f}')

                    self._logger.info(
                        f'Prediction losses ({k}) for {self._steps} steps min: {min:.4f}, mean: {mean:.4f}, max: {max:.4f}, std: {std:.4f}')
                    summary_writer.add_scalars(f'NGE-Validation-{k}/{level}',
                                               {'min': min, 'mean': mean, 'max': max, 'std': std}, e)

        return checkpoint_callback

    def save_history(self):

        files = []
        for level, measures in self._history.items():
            for measure, results in measures.items():
                dataframe = pd.DataFrame(results)
                filename = f'{level}_{measure}.csv'
                dataframe.to_csv(filename, header=True)
                files.append(filename)

        return files