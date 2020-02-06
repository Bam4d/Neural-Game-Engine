import argparse
import logging

import gym
from gym.utils.play import play

from nge_gym.environment_loader import EnvironmentLoader

parser = argparse.ArgumentParser(description='')
parser.add_argument('-i', '--experiment-id', required=True, help='Id of the experiment that contains the model to play')
parser.add_argument('-c', '--experiment-category', required=True, help='Category of the experiment that contains the model to play')


def callback(obs_t, obs_tp1, action, rew, done, info):
    #if rew > 0:
    print(f'Reward:{rew} Done:{done}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()

    category = args.moderage_data_category
    id = args.moderage_data_id

    env_loader = EnvironmentLoader()

    # Load the model and initial states we will use for neural-game-engine
    model, data, levels = env_loader.load_environments(id, category)


    play(gym.make(levels[3]),  fps=10, zoom=4, callback=callback)