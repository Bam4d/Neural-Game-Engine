import gym
import numpy as np
import torch

from training.environment.subprocenv.util import tile_images


class NeuralGameEngineGym(gym.Env):

    def __init__(self, model=None, steps=1, initial_observation=np.zeros(1), sprite_size=10, num_actions=0, action_mapping=None,
                 device='cpu', trace=False):

        self._model = model

        self.action_space = gym.spaces.Discrete(num_actions)
        self._num_actions = num_actions
        self._action_mapping = action_mapping
        self._steps = steps
        self._sprite_size = sprite_size

        self._observation = None

        self._device = device
        self._window = None

        self._initial_observation = None

        self._trace = trace

        if initial_observation is not None:
            self.seed(initial_observation)

    def _forward(self, batch):

        t_batch = self._model.prepare_batch(batch)

        t_actions = t_batch['input_action_batch']
        t_observations_batch = t_batch['input_observation_batch']

        t_predicted_observation_batch, _ = self._model.forward(t_observations_batch, actions=t_actions,
                                                               steps=self._steps, trace=self._trace)

        observation_predictions = t_predicted_observation_batch['observation_predictions']
        reward_predictions = t_predicted_observation_batch['reward_predictions']
        predicted_observations_batch = observation_predictions.cpu().detach().numpy()
        reward_predictions_batch = torch.log_softmax(reward_predictions, 1).cpu().detach().numpy()

        self._observation = np.swapaxes(predicted_observations_batch * 255, 3, 1).astype(np.uint8)
        self._reward = np.squeeze(np.packbits(np.argmax(reward_predictions_batch, axis=1), 1), 1)

    def step(self, action):

        assert self._initial_observation is not None, 'This environment has not been supplied an initial state, either ' \
                                                      'register the environment with an initial_state_file parameter set, ' \
                                                      'or use the seed() function on this environment to provide an initial state'

        if self._observation is None:
            self.reset()

        # (batch, action)
        if isinstance(action, int):
            action = np.expand_dims(np.array(action), 0)

        inputs = np.swapaxes(self._observation / 255.0, 3, 1)

        batch = {
            'input_observation_batch': inputs,
            'input_action_batch': action
        }

        self._forward(batch)

        return self._observation, self._reward, False, None

    def reset(self):

        assert self._initial_observation is not None, 'This environment has not been supplied an initial state, either ' \
                                                      'register the environment with an initial_state_file parameter set, ' \
                                                      'or use the seed() function on this environment to provide an initial state'

        # (batch, channels, width, height)
        if len(self._initial_observation.shape) == 3:
            inputs = np.expand_dims(self._initial_observation, 0)
        else:
            inputs = self._initial_observation

        inputs = np.swapaxes(inputs / 255.0, 3, 1)

        # (batch, action)
        actions = np.zeros(inputs.shape[0], dtype=np.int)

        batch = {
            'input_observation_batch': inputs,
            'input_action_batch': actions
        }

        self._forward(batch)

        return self._observation, 0, False, None

    def render(self, mode='human'):

        assert self._initial_observation is not None, 'This environment has not been supplied an initial state, either ' \
                                                      'register the environment with an initial_state_file parameter set, ' \
                                                      'or use the seed() function on this environment to provide an initial state'

        if mode == 'human':
            if self._window is None:
                self._pyglet = __import__('pyglet')
                self._gl = self._pyglet.gl
                self._window = self._pyglet.window.Window(width=self._width, height=self._height, vsync=False,
                                                          resizable=True)


            tiled_image = tile_images(self._observation)

            image = self._pyglet.image.ImageData(tiled_image.shape[1],
                                                 tiled_image.shape[0],
                                                 'RGB',
                                                 tiled_image.tobytes(),
                                                 pitch=tiled_image.shape[1] * -3
                                                 )



            texture = image.get_texture()
            texture.width = self._width
            texture.height = self._height
            self._window.clear()
            self._gl.glTexParameteri(self._gl.GL_TEXTURE_2D, self._gl.GL_TEXTURE_MAG_FILTER, self._gl.GL_NEAREST)
            self._gl.glTexParameteri(self._gl.GL_TEXTURE_2D, self._gl.GL_TEXTURE_MIN_FILTER, self._gl.GL_NEAREST)
            self._window.switch_to()
            self._window.dispatch_events()
            texture.blit(0, 0)  # draw
            self._window.flip()

        elif mode == 'rgb_array':
            return self._observation[0]

    def close(self):
        super().close()

    def seed(self, seed=None):
        self._initial_observation = seed

        if len(self._initial_observation.shape) == 3:
            observation_shape = self._initial_observation.shape
        else:
            observation_shape = self._initial_observation.shape[1:]

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=observation_shape, dtype=np.uint8)
        self._observation_shape = observation_shape

        self._height = observation_shape[0] * 5
        self._width = observation_shape[1] * 5

        self.reset()

    def get_keys_to_action(self):

        keymap = {
            'LEFT': 'a',
            'UP': 'w',
            'RIGHT': 'd',
            'DOWN': 's',
            'USE': ' '
        }

        return {(ord(f'{keymap[action_name]}'),): action_value for action_name, action_value in
                self._action_mapping.items()}
