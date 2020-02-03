from collections import defaultdict

from gvgai import get_games_path
from gvgai.client.types import Action


import nge_gym
from collections import Counter
import numpy as np
import random
import gym

from training.environment.data_generator import DataGenerator


def get_game_levels(game):
    return [f'gvgai-{game}-lvl{l}-v0' for l in range(5)]


class GVGAIDataGenerator(DataGenerator):

    def __init__(self, name, initial_level_env_level):
        super().__init__(name)

        # gvgai has a sprite size of 10
        self._sprite_size = 10

        self._env = gym.make(initial_level_env_level)
        self._n_actions = self._env.action_space.n
        self._actions = self._env.unwrapped.get_action_meanings()

        self._actions.sort(key=lambda x: x.value)

        self._n_actions = self._env.action_space.n
        self._actions = self._env.unwrapped.get_action_meanings()

    def get_random_gvgai_action(self):
        return self._actions[np.random.randint(self._n_actions)].value

    def get_action_idx(self, action):
        for i, a in enumerate(self._actions):
            if a.value == action:
                return i

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        self._env.stop()

    def get_num_actions(self):
        return self._n_actions

    def get_action_mapping(self):

        mapping = {}
        for i, action in enumerate(self._actions):
            if action == Action.ACTION_UP:
                mapping['UP'] = i
            elif action == Action.ACTION_DOWN:
                mapping['DOWN'] = i
            elif action == Action.ACTION_LEFT:
                mapping['LEFT'] = i
            elif action == Action.ACTION_RIGHT:
                mapping['RIGHT'] = i
            elif action == Action.ACTION_USE:
                mapping['USE'] = i

        return mapping


class GVGAILevelDataGenerator(GVGAIDataGenerator):
    """
    Use the GVGAI nge_gym environment to generate data for learning
    """

    def __init__(self, envs):
        super().__init__('GVGAI', envs[0])

        self._envs = envs

    def get_generator_params(self):
        return {
            'train': self._envs,
            'num_actions': self.get_num_actions()
        }

    def generate_samples(self, batch_size):

        num_batches = len(self._envs)
        self._logger.debug(f'Generating {num_batches} batches with batch size: {batch_size}')

        batches = []

        for environment_id in self._envs:

            self._env.reset(environment_id=environment_id)
            observation_shape = self._env.observation_space.shape
            action_space = self._env.action_space
            if self._n_actions is not None:
                assert self._n_actions == action_space.n, 'Cannot have environments with different action spaces in the same data generator'

            self._n_actions = action_space.n

            self._logger.debug(f'Generating {batch_size} steps in environment {self._env}')
            self._logger.debug(f'Observation Space: {observation_shape}')
            self._logger.debug(f'Action Space: {self._n_actions}')

            observations = []
            actions = []
            rewards = []

            # have 2 add two to the batch size because first image is always blank
            for b in range(batch_size + 1):
                # Random actions so there is wide spread of information
                action = self.get_random_gvgai_action()
                action_idx = self.get_action_idx(action)
                observation, reward, done, _ = self._env.step(action)

                if observation is None:
                    observation = np.zeros(observation_shape)

                # make RGB values between 0 and 1
                observation = observation / 255.0

                # Have to swap axis here to get (channel, width, height) output
                observations.append(np.swapaxes(observation, 0, 2))
                actions.append(action_idx)
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


class GVGAIRandomGenerator(GVGAIDataGenerator):

    def __init__(self, game_name, generate_symmetries=True):
        super().__init__('NGE Learner', f'gvgai-{game_name}-custom-v0')

        self._game_name = game_name
        self._game_location = f'{get_games_path()}/{game_name}_v0'

        self._generate_symmetries = generate_symmetries

        game_level_stats = self._create_level_stats()

        self._level_configs = self._create_level_configs(game_level_stats)

        if self._generate_symmetries:
            self._n_envs = 8
        else:
            self._n_envs = 1


    def _create_level_configs(self, game_level_stats):

        # Create random training level configs
        train_configs = []

        if game_level_stats['level']['consistent_rows'] and game_level_stats['level']['consistent_cols']:
            min_height = game_level_stats['level']['rows']
            max_height = game_level_stats['level']['rows']

            min_width = game_level_stats['level']['cols']
            max_width = game_level_stats['level']['cols']

        else:
            min_height = game_level_stats['level']['rows_mean']
            max_height = game_level_stats['level']['rows_mean'] * 2

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

        game_description = self._get_game_description()
        game_levels = self._get_level_descriptions()

        tile_types = defaultdict(set)
        tile_averages = defaultdict(lambda: 0.0)
        tile_prob = defaultdict(lambda: 0.0)
        tile_counter = Counter()
        edge_counter = Counter()

        level_facets = {}

        level_sizes = []
        for level in game_levels:

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

        num_levels = len(game_levels)

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
            tile_types[edge_char].add('edge')
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
                tile_types[tile].add('singleton')
            else:
                tile_types[tile].add('sparse')
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

    def _get_game_description(self):
        with open(f'{self._game_location}/{self._game_name}.txt') as file:
            return file.readlines()

    def _get_level_descriptions(self):
        levels = []
        for lvl in range(5):
            with open(f'{self._game_location}/{self._game_name}_lvl{lvl}.txt') as file:
                levels.append(file.read().splitlines())
        return levels

    def get_generator_params(self):
        return {
            'train_configs': self._level_configs,
            'num_actions': self.get_num_actions()
        }

    def _rotate_actions90(self, actions):
        rotated_actions = []
        for action in actions:
            if action == 1:  # UP
                rotated_actions.append(2)  # LEFT
            elif action == 2:  # LEFT
                rotated_actions.append(3)  # DOWN
            elif action == 3:  # DOWN
                rotated_actions.append(4)  # RIGHT
            elif action == 4:  # RIGHT
                rotated_actions.append(1)  # UP
            else:
                rotated_actions.append(action)
        return rotated_actions

    def _flip_actions_vert(self, actions):
        flipped_actions = []
        for action in actions:
            if action == 1:  # UP
                flipped_actions.append(3)  # DOWN
            elif action == 3:  # DOWN
                flipped_actions.append(1)  # UP
            else:
                flipped_actions.append(action)
        return flipped_actions

    def _generate_actions(self, batch_size, generate_symmetries=False):

        action_keys = [act.name for act in self._actions]

        has_all_directions = "ACTION_DOWN" in action_keys and "ACTION_LEFT" in action_keys and "ACTION_RIGHT" in action_keys and "ACTION_UP" in action_keys

        symmetric_actions = []

        actions = [self.get_random_gvgai_action() for n in range(batch_size)]

        if not generate_symmetries:
            return [actions]

        if not has_all_directions:
            return [actions]

        for flips in range(2):
            for rots in range(4):
                symmetric_actions.append(actions)
                actions = self._rotate_actions90(actions)
            actions = self._flip_actions_vert(actions)

        return symmetric_actions

    def _get_action_name(self, action):
        if action == 0:
            return 'none'
        if action == 1:
            return 'up'
        if action == 2:
            return 'left'
        if action == 3:
            return 'down'
        if action == 4:
            return 'right'

    def generate_samples(self, batch_size, test=None):
        batches = []


        for config in self._level_configs:

            levels = self.generate_level_data(config, generate_symmetries=self._generate_symmetries)
            actions = self._generate_actions(batch_size + 1,
                                             generate_symmetries=self._generate_symmetries)

            symmetric_batches = defaultdict(lambda: {
                'observations': [],
                'actions': [],
                'rewards': [],
            })

            observation_shape = self._env.observation_space.shape
            self._logger.debug(f'Generating {batch_size} steps in environment {self._env}')
            self._logger.debug(f'Observation Space: {observation_shape}')
            self._logger.debug(f'Action Space: {self._n_actions}')

            self._env.reset(level_data=levels)

            for b in range(batch_size + 1):
                action = actions[:, b]

                observation, reward, done, _ = self._env.step(action.tolist())

                # If the first step is done == True then this is a bad Env and we should rebuild it
                if b == 0 and done.any():
                    return self.generate_samples(batch_size, test)

                for n in range(self._n_envs):
                    observation_n = observation[n] / 255.0
                    observation_n = np.swapaxes(observation_n, 0, 2)
                    symmetric_batches[n]['observations'].append(observation_n)
                    symmetric_batches[n]['actions'].append(self.get_action_idx(action[n]))
                    symmetric_batches[n]['rewards'].append(reward[n])

            for k, symmetric_batch in symmetric_batches.items():
                osbervations_batch = np.stack(symmetric_batch['observations'])

                input_observation_batch = osbervations_batch[:-1]
                expected_observation_batch = osbervations_batch[1:]
                expected_reward_batch = np.stack(symmetric_batch['rewards'])[1:]
                input_action_batch = np.stack(symmetric_batch['actions'])[1:]

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
    def generate_level_data(config, generate_symmetries=False):

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
                    row_string_array.append(GVGAIRandomGenerator._get_sprite(sparse_tile_list))

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

        if not generate_symmetries:
            return ['\n'.join([''.join(r) for r in level_string_array])]

        symmetrical_levels = []

        symmetrical_level = np.array(level_string_array)

        for flips in range(2):
            for rots in range(4):
                symmetrical_levels.append(symmetrical_level)
                symmetrical_level = GVGAIRandomGenerator._rotate_90(symmetrical_level)

            symmetrical_level = GVGAIRandomGenerator._flip_horizontally(symmetrical_level)

        return ['\n'.join([''.join(r) for r in symmetrical_level]) for symmetrical_level in symmetrical_levels]
