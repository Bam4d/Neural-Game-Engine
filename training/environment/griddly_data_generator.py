import random
from collections import Counter
from collections import defaultdict
import yaml
import gym
import numpy as np
from griddly import GymWrapperFactory, gd, GriddlyLoader
from griddly.RenderTools import RenderWindow

from training.environment.data_generator import DataGenerator

class GriddlyDataGenerator(DataGenerator):

    def __init__(self, name, env):
        super().__init__(name)

        self._env = env

        self._env.reset()

        self._n_actions = 5

    def get_num_actions(self):
        return self._n_actions

    def get_image_channels(self):
        return self._env.observation_space.shape[0]

    def get_action_mapping(self):
        return self._env.available_action_input_mappings()


class GriddlyLevelDataGenerator(GriddlyDataGenerator):
    """
    Use the GVGAI nge_gym environment to generate data for learning
    """

    def __init__(self, game_name, gdy_file, tile_size=24, use_vector_observer=False):

        self._render_window = RenderWindow(500, 500)

        self._game_name = game_name
        self._use_vector_observer = use_vector_observer

        wrapper = GymWrapperFactory()
        wrapper.build_gym_from_yaml(game_name,
                                    gdy_file,
                                    player_observer_type=gd.ObserverType.SPRITE_2D if not use_vector_observer else gd.ObserverType.VECTOR,
                                    global_observer_type=gd.ObserverType.SPRITE_2D,
                                    level=0,
                                    tile_size=tile_size)

        loader = GriddlyLoader()
        gdy = loader.load_gdy(gdy_file)

        self._levels = gdy["Environment"]["Levels"]

        self._num_levels = len(self._levels)

        env = gym.make(f'GDY-{game_name}-v0')

        super().__init__(f'Griddly {game_name}. {self._num_levels} Levels', env)

    def get_num_levels(self):
        return self._num_levels

    def get_generator_params(self):
        return {
            'game_name': self._game_name,
            'num_actions': self.get_num_actions()
        }

    def generate_samples(self, batch_size):

        num_batches = len(self._game_name)
        self._logger.debug(f'Generating {num_batches} batches with batch size: {batch_size}')

        batches = []

        for level_id in range(self._num_levels):

            self._env.reset(level_id=level_id)
            observation_shape = self._env.observation_space.shape
            action_space = self._env.action_space

            self._n_actions = 5

            self._logger.debug(f'Generating {batch_size} steps in environment {self._env}')
            self._logger.debug(f'Observation Space: {observation_shape}')
            self._logger.debug(f'Action Space: {self._n_actions}')

            observations = []
            actions = []
            rewards = []

            # have 2 add two to the batch size because first image is always blank
            for b in range(batch_size + 1):
                action = action_space.action_space_dict[action_space.action_names[0]].sample()[0]
                observation, reward, done, _ = self._env.step(action)

                self._render_window.render(observation)

                if not self._use_vector_observer:
                    observation = observation / 255.0

                # Have to swap axis here to get (channel, width, height) output
                observations.append(observation)
                actions.append(action)
                rewards.append(reward)

                if done:
                    self._env.reset()

            np_observations = np.stack(observations)
            np_rewards = np.stack(rewards)

            input_observation_batch = np_observations[:-1]
            expected_observation_batch = np_observations[1:]
            expected_reward_batch = np_rewards[1:]
            input_action_batch = np.stack(actions)[1:]

            batches.append(
                {
                    'input_observation_batch': input_observation_batch,
                    'input_action_batch': input_action_batch,
                    'expected_observation_batch': expected_observation_batch,
                    'expected_reward_batch': expected_reward_batch,
                }
            )

        return batches


class GriddlyRandomGenerator(GriddlyDataGenerator):

    def __init__(self, game_name, gdy_file, tile_size=24, use_vector_observer=False):

        self._game_name = game_name
        self._use_vector_observer = use_vector_observer

        env_name = f'{game_name}-random'
        wrapper = GymWrapperFactory()
        wrapper.build_gym_from_yaml(
            env_name,
            gdy_file,
            player_observer_type=gd.ObserverType.SPRITE_2D if not use_vector_observer else gd.ObserverType.VECTOR,
            global_observer_type=gd.ObserverType.SPRITE_2D,
            level=0,
            tile_size=tile_size)

        loader = GriddlyLoader()
        gdy = loader.load_gdy(gdy_file)

        self._levels = gdy["Environment"]["Levels"]
        self._level_configs = self._create_level_configs()

        env = gym.make(f'GDY-{env_name}-v0')

        super().__init__('Griddly Random Level Generator', env)

    def _create_level_configs(self):

        game_level_stats = self._create_level_stats()

        # Create random training level configs
        train_configs = []

        if game_level_stats['level']['consistent_rows']:
            min_height = game_level_stats['level']['rows']
            max_height = game_level_stats['level']['rows']
        else:
            min_height = game_level_stats['level']['rows_mean']
            max_height = game_level_stats['level']['rows_mean'] * 2

        if game_level_stats['level']['consistent_cols']:
            min_width = game_level_stats['level']['cols']
            max_width = game_level_stats['level']['cols']
        else:
            min_width = game_level_stats['level']['cols_mean']
            max_width = game_level_stats['level']['cols_mean'] * 2

        train_configs.append(
            {
                'min_width': min_width,
                'max_width': max_width,
                'min_height': min_height,
                'max_height': max_height,
                'tiles': game_level_stats['tiles']
            }
        )

        return train_configs

    def _create_level_stats(self):

        tile_types = defaultdict(list)
        tile_averages = defaultdict(lambda: 0.0)
        tile_prob = defaultdict(lambda: 0.0)
        tile_counter = Counter()
        edge_counter = Counter()

        level_facets = {}

        level_sizes = []
        for level in self._levels:
            level = level.split('\n')
            rows = len(level)
            cols = len(level[0])

            level_sizes.append((rows, cols))

            for i, row in enumerate(level):
                for j, char in enumerate(row):
                    if char == '\n':
                        break

                    tile_counter[char] += 1
                    if i == 0 or j == 0 or i == rows - 1 or j == cols - 1:
                        edge_counter[char] += 1

        num_levels = len(self._levels)

        row_size_counter = Counter()
        col_size_counter = Counter()

        for level_size in level_sizes:
            row_size_counter[level_size[0]] += 1
            col_size_counter[level_size[1]] += 1

        if len(row_size_counter) == 1:
            level_facets['consistent_rows'] = True
            level_facets['rows'] = list(row_size_counter.keys())[0]
        else:
            level_facets['consistent_rows'] = False
            level_facets['rows_mean'] = np.mean(list(row_size_counter.keys()))

        if len(col_size_counter) == 1:
            level_facets['consistent_cols'] = True
            level_facets['cols'] = list(col_size_counter.keys())[0]
        else:
            level_facets['consistent_cols'] = False
            level_facets['cols_mean'] = np.mean(list(row_size_counter.keys()))

        has_edge = False
        if len(edge_counter) == 1:
            edge_char = list(edge_counter.keys())[0]
            # This tile is always on the edge of the map
            tile_types[edge_char].append('edge')
            has_edge = True

        # if theres a wall around the outside of the grid then calculate probabilities only with respect to stuff inside
        # the wall
        level_size_sum = 0
        edge_counter = 0
        if has_edge:
            for r, c in level_sizes:
                edge_size = (2 * r + 2 * c - 4)
                edge_counter += edge_size
                level_size_sum += (r * c - edge_size)
        else:
            level_size_sum = np.sum([r * c for r, c in level_sizes])

        for tile, count in tile_counter.items():
            tile_average = tile_counter[tile] / num_levels

            # There has to exactly one of these in every level
            if tile_average == 1.0:
                tile_types[tile].append('singleton')
            else:
                tile_types[tile].append('sparse')
                if 'edge' in tile_types[tile]:
                    tile_prob[tile] = (tile_counter[tile] - edge_counter) / level_size_sum
                else:
                    tile_prob[tile] = tile_counter[tile] / level_size_sum

            tile_averages[tile] = tile_average

        tiles = {}
        for tile, types in tile_types.items():
            tile_config = {}
            tile_config['types'] = types
            if tile in tile_prob:
                tile_config['prob'] = tile_prob[tile]

            tiles[tile] = tile_config

        return {'level': level_facets, 'tiles': tiles}

    def get_generator_params(self):
        return {
            'train_configs': self._level_configs,
            'num_actions': self.get_num_actions()
        }

    def _generate_actions(self, batch_size):
        action_space = self._env.action_space
        simple_action_space = self._env.action_space.action_space_dict[action_space.action_names[0]]
        actions = [simple_action_space.sample()[0] for n in range(batch_size)]
        return np.array(actions)

    def generate_samples(self, batch_size, test=None):
        batches = []

        for config in self._level_configs:

            level = self.generate_level_data(config)
            actions = self._generate_actions(batch_size + 1)

            batch_observations = []
            batch_actions = []
            batch_rewards = []

            observation_shape = self._env.observation_space.shape
            self._logger.debug(f'Generating {batch_size} steps in environment {self._env}')
            self._logger.debug(f'Observation Space: {observation_shape}')
            self._logger.debug(f'Action Space: {self._n_actions}')

            self._env.reset(level_string=level)

            for b in range(batch_size + 1):
                action = actions[b]

                observation, reward, done, _ = self._env.step(action)

                if self._use_vector_observer:
                    batch_observations.append(observation)
                else:
                    batch_observations.append(observation / 255.0)
                batch_actions.append(action)
                batch_rewards.append(reward)

            input_observation_batch = np.stack(batch_observations[:-1])
            expected_observation_batch = np.stack(batch_observations[1:])
            expected_reward_batch = np.stack(batch_rewards[1:])
            input_action_batch = np.stack(batch_actions[1:])

            batches.append(
                {
                    'input_observation_batch': input_observation_batch,
                    'input_action_batch': input_action_batch,
                    'expected_observation_batch': expected_observation_batch,
                    'expected_reward_batch': expected_reward_batch,
                }
            )

        return batches

    @staticmethod
    def _get_sprite(sorted_sparse_tile_list):
        sprite_select = random.uniform(0, 1)

        # Assuming that probability of sorted tiles sums to 1
        cumulative_prob = 0.0
        for sparse_tile in sorted_sparse_tile_list:
            cumulative_prob += sparse_tile[1]
            if sprite_select < cumulative_prob:
                return sparse_tile[0]

    @staticmethod
    def _flip_horizontally(level_data):
        return np.flip(level_data, axis=0)

    @staticmethod
    def _rotate_90(level_data):
        return np.rot90(level_data)

    @staticmethod
    def generate_level_data(config):

        tile_config = config['tiles']

        sparse_tile_list = [(tile, properties['prob']) for tile, properties in tile_config.items() if
                            'prob' in properties]

        singleton_tile_list = [(tile, properties) for tile, properties in tile_config.items() if
                               'singleton' in properties['types']]

        edge_tile = None
        for tile, properties in tile_config.items():
            if 'edge' in properties['types']:
                edge_tile = tile

        total_probs = np.sum([tile[1] for tile in sparse_tile_list])

        assert total_probs <= 1.0, 'Probabilities must not sum larger than 1'

        scale = 1 / total_probs

        sparse_tile_list = [(tile[0], tile[1] * scale) for tile in sparse_tile_list]

        width = np.random.randint(config['min_width'], config['max_width'] + 1)
        height = np.random.randint(config['min_height'], config['max_height'] + 1)

        level_string_array = []
        for h in range(height):
            row_string_array = []
            for w in range(width):
                if edge_tile is not None and (w == 0 or h == 0 or h == height - 1 or w == width - 1):
                    row_string_array.append(edge_tile)
                else:
                    row_string_array.append(GriddlyRandomGenerator._get_sprite(sparse_tile_list))

            level_string_array.append(row_string_array)

        taken_singletons = []
        for singleton_tile in singleton_tile_list:
            found = False
            while not found:
                random_coords = random.randint(1, width - 2), random.randint(1, height - 2)
                if random_coords not in taken_singletons:
                    level_string_array[random_coords[1]][random_coords[0]] = singleton_tile[0]
                    taken_singletons.append(random_coords)
                    found = True

        return '\n'.join([''.join(r) for r in level_string_array])
