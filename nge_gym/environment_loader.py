import torch
import numpy as np
from gym.envs.registration import register
from moderage import ModeRage

class EnvironmentLoader():

    def __init__(self, device=None):
        self._device = device if device is not None else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._mr = ModeRage()

    def register_environment_model(self, game_id, model, steps, num_actions, action_mapping, intial_state_file=None, trace_handler=None):
        model.to(self._device)

        if model._trace_handler is None:
            model._trace_handler = trace_handler

        # Load initial states
        initial_observation = np.load(intial_state_file) if intial_state_file is not None else None

        model.eval()

        register(
            id=game_id,
            entry_point='nge_gym.environment:NeuralGameEngineGym',
            kwargs={
                'model': model,
                'steps': steps,
                'initial_observation': initial_observation,
                'num_actions': num_actions,
                'action_mapping': action_mapping,
                'device': self._device,
                'trace': trace_handler is not None
            }
        )

        return model

    def load_model(self, id, category):
        """
        Just loads and returns the environment model
        :param id:
        :param category:
        :return: model, data
        """

        data = self._mr.load(id, category)
        model_file = data.get_file('model.tch')
        model = torch.load(model_file, map_location=self._device)

        return model, data

    def load_environments(self, id, category):
        """
        Loads the environment model and levels from the initial conditions
        :param id:
        :param category:
        :param game_id:
        :param initial_state_file:
        :return:
        """

        model, data = self.load_model_from_moderage(id, category)

        steps = data.meta['automata_steps']
        num_actions = data.meta['num_actions']
        action_mapping = data.meta['action_map']

        initial_state_files = [data.get_file(file['filename']) for file in data.files if file['filename'].endswith('initial.npy')]

        levels = []
        for i, state_files in enumerate(initial_state_files):

            game_id = f'nge-{id}-lvl{i}-v0'

            levels.append(game_id)

            self.register_environment_model(game_id, model, steps, num_actions, action_mapping, state_files)

        return model, data.meta, levels