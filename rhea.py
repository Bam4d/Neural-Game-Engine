import argparse
import logging
import gvgai
import gym
import numpy as np
from RollingHorizonEA import Environment, RollingHorizonEvolutionaryAlgorithm

from nge_gym.environment_loader import EnvironmentLoader


class RollingHorizonNGEEnvironment(Environment):

    def __init__(self, id, category, n_eval, level, max_steps=100, render=False):
        super().__init__("RHEA_NGE")

        self._n_eval = n_eval
        self._render = render

        env_loader = EnvironmentLoader()
        nge_name = f'nge-{id}-v0'

        # Load the model and the initial conditions
        rollout_model, data = env_loader.load_model(id, category)
        env_loader.register_environment_model(nge_name, rollout_model, data.meta['ngpu_iterations'],
                                              data.meta['num_actions'],
                                              data.meta['action_map'])

        self._rollout_env = gym.make(nge_name)

        self._real_env = gym.make(level)
        self._original_observation = self._real_env.reset()
        self._original_terminated = False

        self._action_space = self._real_env.action_space

        self._max_steps = max_steps
        self._steps = 0

        self._score = 0.0

    def perform_action(self, action):
        self._original_observation, original_reward, self._original_terminated, _ = self._real_env.step(action)
        self._real_env.render()
        self._score += original_reward
        self._steps += 1

    def evaluate_rollout(self, solution, discount_factor=0, ignore_frames=0):
        rewards = []
        dones = []

        self._rollout_env.seed(np.tile(self._original_observation, (self._n_eval, 1, 1, 1)))

        for i in range(solution.shape[1]):
            s = np.squeeze(solution[:, i])
            observation, reward, done, _ = self._rollout_env.step(s)

            if self._render:
                self._rollout_env.render()

            rewards.append(reward / (i+1.0))
            dones.append(done)

        # Done is really good for the score
        return np.sum(rewards, axis=0)

    def get_random_action(self):
        return [np.random.randint(self._action_space.n)]

    def is_game_over(self):
        return self._original_terminated or self._steps == self._max_steps

    def get_current_score(self):
        return self._score

    def __del__(self):
        self._real_env.stop()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-i', '--id', required=True, help='Id of the NGE model')
    parser.add_argument('-l', '--level', required=True, help='The GVGAI environment to play, i.e: \'gvgai-sokoban-lvl0-v0\'')
    parser.add_argument('-e', '--evals', type=int, required=True, help='The number of evaluation rollouts')
    parser.add_argument('-r', '--rollout', type=int, required=True, help='The size of rollouts')
    parser.add_argument('-p', '--mutation-probability', type=float, default=0.2, help='The probability of mutation in rollout actions')
    parser.add_argument('-R', '--render-rollouts', dest='render', action='store_true')

    args = parser.parse_args()

    environment_id = args.id
    category = 'NGE_Learner'
    num_evals = args.evals
    rollout_length = args.rollout
    mutation_probability = args.mutation_probability
    render = args.render
    level = args.level

    # Set up the problem domain as one-max problem
    environment = RollingHorizonNGEEnvironment(environment_id, category, num_evals, level, render=render)

    rhea = RollingHorizonEvolutionaryAlgorithm(rollout_length, environment, mutation_probability, num_evals)

    rhea.run()
