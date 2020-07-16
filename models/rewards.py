from torch import nn
import torch.nn.functional as F


class PoolingRewardDecoder(nn.Module):

    def __init__(self, num_actions, state_channels):
        super().__init__()

        self._observation_encoder = nn.Sequential(
            nn.Conv2d(3, state_channels, kernel_size=24, stride=24),
            nn.ReLU(),
            nn.BatchNorm2d(state_channels)
        )

        self._action_encoder = nn.Sequential(
            nn.Conv2d(num_actions, state_channels, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(state_channels)
        )

        self._network = nn.Sequential(
            nn.Conv2d(state_channels, state_channels, kernel_size=3),
            nn.ReLU(),
            nn.BatchNorm2d(state_channels),
            nn.Conv2d(state_channels, state_channels // 2, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(state_channels // 2),
            nn.Conv2d(state_channels // 2, state_channels // 4, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(state_channels // 4),
            nn.Conv2d(state_channels // 4, 16, kernel_size=3),
        )

    # Just get the maximum value from the pooling operation
    def forward(self, observation, actions):
        encoded_observation = self._observation_encoder(observation)
        encoded_actions = self._action_encoder(actions)

        encoded_state = encoded_actions + encoded_observation

        nin = self._network(encoded_state)
        pool = F.max_pool2d(nin, kernel_size=nin.shape[2:])
        return pool.reshape(-1, 2, 8)
