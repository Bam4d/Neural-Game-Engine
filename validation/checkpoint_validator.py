import logging
from collections import defaultdict

import pandas as pd
import numpy as np
from PIL import Image

from training.environment.gvgai_data_generator import GVGAILevelDataGenerator
from validation.prediction_accuracy import PredictionAccuracyMeasure


class CallBackValidator():

    def __init__(self, validation_repeats, validation_steps, ngpu_iterations, levels, level_generator=None):
        self._levels = levels

        self._history = defaultdict(lambda: defaultdict(lambda: []))

        self._repeats = validation_repeats
        self._steps = validation_steps
        self._ngpu_iterations = ngpu_iterations

        self._level_generator = level_generator if level_generator is not None else GVGAILevelDataGenerator(self._levels)

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        self._level_generator.cleanup()

    def get_callback(self, model, summary_writer):
        logger = logging.getLogger("Generalisation")

        num_actions = self._level_generator.get_num_actions()
        action_mapping = self._level_generator.get_action_mapping()

        if 'level_data' in self._levels[0]:
            compare_levels = [
                {
                    'id': level['id'],
                    'name': level['name'],
                    'level_data': self._level_generator.generate_level_data(level['level_data'], False)[0]
                }
                for level in self._levels
            ]
        else:
            compare_levels = [{'id': level, 'name': level} for level in self._levels]

        self._prediction_accuracy_measure = PredictionAccuracyMeasure(model._model,
                                                                      compare_levels,
                                                                      ngpu_iterations=self._ngpu_iterations,
                                                                      num_actions=num_actions,
                                                                      action_mapping=action_mapping)

        def checkpoint_callback(e):
            # Then print a trace of learning a value
            for i, batch in enumerate(self._level_generator.generate_samples(32)):
                t_batch = model._model.prepare_batch(batch)
                # Only view out the largest
                trace = i == len(self._levels) - 1
                level = self._levels[i]

                (loss, _), predicted = model.eval(t_batch, trace=trace)
                logger.info(f'Loss at {level}: {loss:.4f}')

            results = self._prediction_accuracy_measure.calculate(self._steps, self._repeats)

            for k, prediction_accuracy_error in results.items():

                level_data = []
                for level, data in prediction_accuracy_error.items():
                    level_data.append(data)

                mean = np.mean(level_data)

                self._history[k]['epoch'].append(e)
                self._history[k]['mean'].append(mean)

                logger.info(f'Prediction stat ({k}) for {self._steps} steps mean: {mean:.4f}')

                if k.startswith('tile'):
                    measure = k.split('_')[2]
                    summary_writer.add_scalars(f'NGE-Validation/tile_{measure}_mean', {f'{k}': mean}, e)
                else:
                    summary_writer.add_scalars(f'NGE-Validation/{k}', {'mean': mean}, e)

        return checkpoint_callback

    def save_history(self):

        files = []
        for measure, results in self._history.items():
            dataframe = pd.DataFrame(results)
            filename = f'{measure}.csv'
            dataframe.to_csv(filename, header=True)
            files.append(filename)

        for i, tile in enumerate(self._prediction_accuracy_measure.get_tiles()):
            tile_image = Image.fromarray(tile)
            filename = f'tile_{i}.png'
            tile_image.save(filename)
            files.append(filename)

        return files