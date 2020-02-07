import argparse
import logging

import gym
from gym.utils.play import play

from nge_gym.environment_loader import EnvironmentLoader

parser = argparse.ArgumentParser(description='')
parser.add_argument('-i', '--id', required=True, help='Id of the experiment that contains the model to play')
parser.add_argument('-l', '--level', required=True, type=int, help='The level to play, each model stores the 5 original gvgai levels.')


def callback(obs_t, obs_tp1, action, rew, done, info):
    if rew > 0:
        print(f'Reward:{rew}')


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    args = parser.parse_args()

    id = args.id
    level = args.level

    env_loader = EnvironmentLoader()

    # Load the model and initial states we will use for neural-game-engine
    model, data, levels = env_loader.load_environments(id, 'NGE_Learner')


    play(gym.make(levels[level]), fps=10, zoom=4, callback=callback)