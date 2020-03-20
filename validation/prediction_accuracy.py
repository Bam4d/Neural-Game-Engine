import logging
import uuid
import time
from collections import defaultdict

import gym
import numpy as np

from nge_gym.environment_loader import EnvironmentLoader
from training.util import calc_precision_recall_f1_bacc


class PredictionAccuracyMeasure():

    def __init__(self, model, compare_levels, ngpu_iterations, num_actions, action_mapping):
        self._compare_levels = compare_levels
        self._model = model

        self._logger = logging.getLogger("PredictionAccuracyMeasure")

        self._env_name = f'nge_{uuid.uuid4()}-v0'

        loader = EnvironmentLoader()
        loader.register_environment_model(self._env_name, model, ngpu_iterations, num_actions, action_mapping)

        self._original_env = None

        self._observation_tile_hashes = set()
        self._observation_tiles = []

    def get_tiles(self):
        return self._observation_tiles

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        if self._original_env is not None:
            self._original_env.stop()

    def _abs_diff_uint8(self, img1, img2):
        a = img1 - img2
        b = np.uint8(img1 < img2) * 254 + 1
        return a * b

    def _create_tile_map(self, image, height, width):

        image_tiled = image.reshape(height, 10, width, 10, 3)

        tile_map = np.zeros((height, width))
        # Second pass to create tile map
        for x in range(width):
            for y in range(height):
                smallest_diff = float("inf")
                closest_tile = -1
                for i, tile in enumerate(self._observation_tiles):
                    obs_tile = image_tiled[y, :, x]
                    diff = np.sum(self._abs_diff_uint8(obs_tile, tile))
                    if smallest_diff > diff:
                        smallest_diff = diff
                        closest_tile = i
                tile_map[y, x] = closest_tile

        return tile_map

    def _compare_observation_tiles_to_nge(self, observation, nge_observation):

        height = observation.shape[0] // 10
        width = observation.shape[1] // 10

        observation_tiled = observation.reshape(height, 10, width, 10, 3)

        # First pass to get all tiles
        for x in range(width):
            for y in range(height):
                tile = observation_tiled[y, :, x]
                tile_hash = hash(tile.data.tobytes())
                if tile_hash not in self._observation_tile_hashes:
                    self._observation_tile_hashes.add(tile_hash)
                    self._observation_tiles.append(tile)

        observation_tile_map = self._create_tile_map(observation, height, width)
        nge_tile_map = self._create_tile_map(nge_observation, height, width)

        tile_stats = {}
        for i, tile in enumerate(self._observation_tiles):
            observation_single_tile_map = observation_tile_map == i
            nge_single_tile_map = nge_tile_map == i
            stats = calc_precision_recall_f1_bacc(nge_single_tile_map, observation_single_tile_map)
            tile_stats[i] = np.array(stats)

        return tile_stats

    def _collect_data(self, steps, repeats):

        for original_level in self._compare_levels:

            self._logger.info(
                f'Collecting data from NGE environment and original level [{original_level}]')

            mean_squared_error_repeats = []
            tile_stats_repeats = []

            for repeat in range(repeats):

                self._logger.info(
                    f'Generating data for [{self._env_name}] and gvgai level [{original_level["name"]}], repeat: {repeat}')

                gym_params = dict(original_level)
                del gym_params['name']

                nge_env = gym.make(self._env_name)

                if self._original_env is None:
                    self._original_env = gym.make(**gym_params)
                    self._original_env.reset()
                else:
                    self._original_env.reset(**gym_params)

                # Have to step more than once before observation works (need to fix this in gvgai)
                original_observation, original_reward, original_done, _ = self._original_env.step(0)

                # Seed the nge environment with the observation from the original environment
                # This is all NGE needs to play the game
                nge_env.seed(original_observation)

                action_desc = sorted(self._original_env.unwrapped.get_action_meanings(), key=lambda k: k.value)

                # assert nge_env.action_space == self._original_env.action_space
                assert nge_env.observation_space == self._original_env.observation_space

                # nge_env.render()
                # original_env.render()

                mean_squared_error_steps = []
                tile_stats_steps = []

                for s in range(steps):
                    # Produce a random action
                    action_id = np.random.randint(self._original_env.action_space.n)
                    action = action_desc[action_id].value

                    # Get the observations from both environments after the action
                    nge_observation, nge_reward, nge_done, _ = nge_env.step(action_id)
                    original_observation, original_reward, original_done, _ = self._original_env.step(action)

                    nge_observation = np.squeeze(np.uint8(nge_observation))
                    original_observation = np.squeeze(np.uint8(original_observation))

                    diff = self._abs_diff_uint8(nge_observation, original_observation)

                    squared_error = np.square(diff / 255.0)

                    tile_stats = self._compare_observation_tiles_to_nge(original_observation, nge_observation)

                    mean_squared_error_steps.append(np.mean(squared_error))
                    tile_stats_steps.append(tile_stats)

                    if original_done:
                        self._original_env.reset()
                        nge_env.reset()

                mean_squared_error_repeats.append(np.stack(mean_squared_error_steps))
                tile_stats_repeats.append(tile_stats_steps)

            num_tiles = len(self._observation_tiles)

            level_tile_stats = np.zeros((repeats, steps, num_tiles, 4))

            for r in range(repeats):
                for s in range(steps):
                    for t in range(num_tiles):
                        if t in tile_stats_repeats[r][s]:
                            stat = tile_stats_repeats[r][s][t]
                        else:
                            stat = [0.0, 0.0, np.nan, 0.5]
                        level_tile_stats[r,s,t] = stat

            tile_stats_data = {}
            for tile_idx in range(num_tiles):
                for m, measure in enumerate(['p', 'r', 'f1', 'bacc']):
                    tile_stats_data[f'tile_{tile_idx}_{measure}'] = np.nanmean(level_tile_stats[:, :, tile_idx, m], axis=0)

            yield {
                'steps': steps,
                'repeats': repeats,
                'nge_level': self._env_name,
                'original_level': original_level,
                'test_data': {
                    'observation_mean_squared_error': np.mean(mean_squared_error_repeats, axis=0),
                    **tile_stats_data
                }
            }

    def calculate(self, steps, repeats):

        all_test_data = defaultdict(lambda: {})

        for level_comparison_data in self._collect_data(steps, repeats):
            # mean squared error of prediction of frames over time compared to real gym level

            for test, data in level_comparison_data['test_data'].items():
                all_test_data[test] = {
                    **all_test_data[test],
                    level_comparison_data["original_level"]["name"]: data,
                }

        return all_test_data