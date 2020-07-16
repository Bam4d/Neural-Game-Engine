import numpy as np
from training.environment.data_generator import DataGenerator


def tile_actions(action_batch, width, height, sprite_size=24, num_actions=5):
    oh_action = DataGenerator.convert_to_one_hot(action_batch, num_actions)

    # Tile the action batches to the same shape as the cgru state
    return np.tile(
        oh_action.reshape(
            oh_action.shape[0],
            oh_action.shape[1],
            1,
            1
        ),
        (
            1,
            1,
            width // sprite_size,
            height // sprite_size
        )
    )


def binarize_rewards(reward_batch):
    return np.unpackbits(np.expand_dims(reward_batch, 1), axis=1)