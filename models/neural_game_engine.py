import logging
import numpy as np
import torch
from torch import nn

from models.ngpu import CGRUCell
from models.rewards import PoolingRewardDecoder
from models.utils import tile_actions, binarize_rewards


class NeuralGameEngine(nn.Module):

    def __init__(self, state_channels, reward_state_channels, num_actions, observation_noise_std=None,
                 saturation_limit=0.9, trace_handler=None, summary_writer=None):
        super().__init__()
        self._logger = logging.getLogger("Neural Game Engine")

        self._summary_writer = summary_writer
        self._trace_handler = trace_handler

        self._num_actions = num_actions

        self._observation_noise_std = observation_noise_std

        self._observation_encoder = nn.Sequential(
            nn.Conv2d(3, state_channels, kernel_size=10, stride=10),
        )

        self._observation_decoder = nn.Sequential(
            nn.ConvTranspose2d(state_channels, state_channels // 2, kernel_size=10, stride=10),
            nn.ReLU(),
            nn.ConvTranspose2d(state_channels // 2, 3, 1),
            nn.Sigmoid()
        )

        self._action_encoder = nn.Sequential(
            nn.Conv2d(num_actions, state_channels, 1),
        )

        self._reward_forward = PoolingRewardDecoder(num_actions, reward_state_channels)

        self._cgru_cell = CGRUCell(state_channels, saturation_limit=saturation_limit)

        # self._summary_writer = None
        self._train_step = 0
        self._bins = np.arange(0, 10, 0.1)

        self._logger.info(f'Using Neural NGPU engine with selective gating')

    def prepare_batch(self, batch_data):
        t_batch = {}
        for variable_name, batch in batch_data.items():
            # If we are using local models, we tile the actions
            if variable_name == 'input_action_batch':
                width = batch_data['input_observation_batch'].shape[2]
                height = batch_data['input_observation_batch'].shape[3]

                batch = tile_actions(batch, width, height, num_actions=self._num_actions)

            if variable_name == 'expected_reward_batch':
                # Convert to 8 bit binary number
                batch = binarize_rewards(batch.astype(np.uint8))

            t_batch[variable_name] = torch.FloatTensor(batch).to(self._device)

        return t_batch

    def forward(self, input, actions=None, steps=None, trace=False):
        # Input will be in form (batch_size, channels, width, height)

        # If steps is not defined, then steps = input.shape[1]
        steps = steps if steps is not None else input.shape[3]

        if self.training and self._observation_noise_std is not None:
            input = torch.clamp(input + torch.randn(input.shape).to(self._device) * self._observation_noise_std,
                                min=0.0, max=1.0)

        # Calculate initial state
        initial_state = self._observation_encoder(input)

        if trace and self._summary_writer is not None:
            self._summary_writer.add_histogram("encoded_visual_input", initial_state, global_step=self._train_step,
                                               bins=self._bins)

        # If we have actions we want to combine them with the
        if actions is not None:
            embedded_actions = self._action_encoder(actions)
            initial_state = initial_state + embedded_actions

        if trace and self._summary_writer is not None:
            self._summary_writer.add_histogram("action_conditioned_visual_input", initial_state,
                                               global_step=self._train_step,
                                               bins=self._bins)

        final_state, saturation_cost = self._cgru(initial_state, steps, trace)

        if self.training:
            self._train_step += 1

        observation_prediction = self._observation_decoder(final_state)
        reward_prediction = self._reward_forward(input.detach(), actions)

        outputs = {
            'observation_predictions': observation_prediction,
            'reward_predictions': reward_prediction
        }

        # calculate the output state
        return outputs, saturation_cost

    def _cgru(self, state, steps, trace=False):
        saturation_costs = []
        for i in range(steps):

            # Update the state
            state, saturation_cost = self._cgru_cell(state)

            # If the trace is being tracked in any way (rendered, sent to tensorboard etc)
            if trace and self._trace_handler is not None:
                output = self._observation_decoder(state)
                self._trace_handler.trace(i, steps, output, state)

            saturation_costs.append(saturation_cost)

        return state, torch.sum(torch.stack(saturation_costs))

    def saveable(self):
        """
        A hack so this module can be saved,
        python cannot pickle module objects, so have to set the trace handler / summary writer to None
        :return:
        """
        self._summary_writer = None
        self._trace_handler = None
        return self

    def to(self, *args, **kwargs):
        self._device = args[0]
        return super().to(*args, **kwargs)
