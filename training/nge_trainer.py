import numpy as np
import pandas as pd
import torch
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import Adamax
from torch.optim.lr_scheduler import ReduceLROnPlateau

from models.neural_game_engine import NeuralGameEngine
from training.loss_components import LossComponentCollector
from training.trainable import Trainable
from training.util import calc_precision_recall_f1_bacc


class GymLearner(Trainable):

    def __init__(self, hyperparameters, data_generator, initial_state_generator=None, trace_handler=None,
                 summary_writer=None):
        super().__init__('NGE_Learner', moderage_category=None, moderage_data_id=None, summary_writer=summary_writer)

        self._hyperparameters = hyperparameters
        self._data_generator = data_generator

        self._state_channels = hyperparameters['state_channels']

        self._saturation_cost_weight = hyperparameters['saturation_cost_weight']
        self._saturation_limit = hyperparameters['saturation_limit']

        self._gradient_clip = hyperparameters['gradient_clip']

        self._observation_noise_std = hyperparameters['observation_noise_std']

        self._reward_loss_coeff = hyperparameters['reward_loss_coeff']
        self._reward_state_channels = hyperparameters['reward_state_channels']
        self._reward_class_weight = hyperparameters['reward_class_weight']

        self._state_channels = hyperparameters['state_channels']
        self._batch_size = hyperparameters['batch_size']

        self._learning_rate_patience = hyperparameters['learning_rate_patience']
        self._learning_rate_decay_factor = hyperparameters['learning_rate_decay_factor']

        self._iterations = hyperparameters['ngpu_iterations']

        self._num_actions = data_generator.get_num_actions()
        self._initial_state_generator = initial_state_generator

        self._model = NeuralGameEngine(
            self._state_channels,
            self._reward_state_channels,
            self._num_actions,
            observation_noise_std=self._observation_noise_std,
            saturation_limit=self._saturation_limit,
            trace_handler=trace_handler,
            summary_writer=summary_writer,
        ).to(self._device)

        self._optimizer = Adamax(self._model.parameters(), lr=hyperparameters['learning_rate'])

        if self._learning_rate_patience is not None:
            self._scheduler = ReduceLROnPlateau(self._optimizer, mode='min', factor=self._learning_rate_decay_factor,
                                                verbose=True,
                                                patience=self._learning_rate_patience)

        self._mse_observation_loss_criterion = MSELoss().to(self._device)
        self._ce_reward_loss_criterion = CrossEntropyLoss(weight=torch.tensor(self._reward_class_weight)).to(
            self._device)

        self._logger.info('Created Automata Learner')
        self._logger.info(f'Data Generator: {data_generator.get_name()}')
        self._logger.info(f'State channels: {self._state_channels}')

    def is_training(self):
        return self._model.training

    def _get_lr(self):
        for param_group in self._optimizer.param_groups:
            return param_group['lr']

    def _loss(self, predictions, t_batch, saturation_cost=None):

        observation_targets = t_batch['expected_observation_batch']
        observation_predictions = predictions['observation_predictions']

        reward_targets = t_batch['expected_reward_batch']
        reward_predictions = predictions['reward_predictions']

        batch_size = observation_targets.shape[0]

        loss_components = {}

        # Calculate mean square error loss component
        mse_observation_loss = self._mse_observation_loss_criterion(observation_predictions, observation_targets)
        loss_components['mse_observation_loss'] = mse_observation_loss

        # Calculate cross entropy loss for reward
        reward_target_class = reward_targets.type(torch.long)
        ce_reward_loss = self._ce_reward_loss_criterion(reward_predictions, reward_target_class)


        reward_predictions_np = np.argmax(reward_predictions.detach().cpu().numpy(), axis=1)
        reward_target_np = reward_target_class.detach().cpu().numpy()

        reward_precision, reward_recall, reward_f1, reward_bacc = calc_precision_recall_f1_bacc(reward_predictions_np,
                                                                                                reward_target_np)

        # Calculate saturation cost loss
        loss_components['saturation_loss'] = saturation_cost * self._saturation_cost_weight

        total_loss = torch.sum(torch.stack([loss for _, loss in loss_components.items()]))

        loss_components['ce_reward_loss'] = ce_reward_loss
        total_loss += ce_reward_loss * self._reward_loss_coeff

        detached_loss_components = {k: loss.detach().cpu().numpy() for k, loss in loss_components.items()}

        detached_loss_components['reward_precision'] = reward_precision
        detached_loss_components['reward_recall'] = reward_recall
        detached_loss_components['reward_bacc'] = reward_bacc
        detached_loss_components['reward_f1'] = reward_f1


        reward_rate = (reward_targets.detach().cpu().numpy().sum(axis=1) > 0).sum() / batch_size

        detached_loss_components['reward_rate'] = reward_rate

        return total_loss, detached_loss_components

    def forward(self, t_batch, steps=1, trace=False):

        inputs = t_batch['input_observation_batch']
        actions = t_batch['input_action_batch']

        return self._model.forward(inputs, actions=actions, steps=steps, trace=trace)

    def train_batches(self):
        training_batches = self._data_generator.generate_samples(batch_size=self._batch_size)

        train_batch_losses = []
        loss_component_collector = LossComponentCollector()
        for training_batch in training_batches:
            t_prepared_batch = self._model.prepare_batch(training_batch)
            batch_loss, loss_components_batch = self.train_batch(t_prepared_batch)
            train_batch_losses.append(batch_loss)
            loss_component_collector.append_loss_components_batch(loss_components_batch)

        return np.mean(train_batch_losses), loss_component_collector

    def eval(self, t_batch, trace=False):
        # Get predictions
        predictions, saturation_costs = self.forward(
            t_batch, steps=self._iterations, trace=trace)

        # Calculate losses
        loss, loss_components = self._loss(predictions, t_batch,
                                           saturation_costs)

        # Get loss
        loss.backward()

        # Return the loss from the single batch step
        return (loss.data.detach().cpu().numpy(), loss_components), predictions

    def train_batch(self, t_batch):
        # Get predictions
        predictions, saturation_cost = self.forward(t_batch, steps=self._iterations)

        # Calculate losses
        total_loss, loss_components = self._loss(predictions, t_batch, saturation_cost)

        # Update the weights
        self._optimizer.zero_grad()
        total_loss.backward()

        # clip gradient
        torch.nn.utils.clip_grad_norm_(self._model.parameters(), self._gradient_clip)

        self._optimizer.step()

        return total_loss.data.detach().cpu().numpy(), loss_components

    def train(self, training_epochs, checkpoint_callback=None, callback_epoch=10, **kwargs):

        training_mean_loss_component_collector = LossComponentCollector(500)
        for e in range(training_epochs):

            self._epoch = e

            self._model.eval()
            # If we want to do something at specific points during training then we can set a checkpoint callback
            if checkpoint_callback is not None and self._epoch % callback_epoch == 0:
                checkpoint_callback(e)

            self._model.train()
            training_loss, training_loss_components = self.train_batches()

            training_mean_loss_components = training_loss_components.get_means()
            training_mean_loss_component_collector.append_loss_components_batch(training_mean_loss_components)

            debug_string = ', '.join(
                [f'{k}: {v:.4f}' for k, v in training_mean_loss_component_collector.get_window_mean().items()])

            self._logger.info(
                f'Epoch [{e + 1}/{training_epochs}], Lr: {self._get_lr():.4f}, {debug_string}')

            if self._summary_writer is not None:

                for component_key, component_value in training_mean_loss_component_collector.get_window_mean().items():
                    self._summary_writer.add_scalars(f'{self.get_name()}/training/{component_key}',
                                                     {component_key: component_value}, e)

            if self._learning_rate_patience is not None:
                self._scheduler.step(training_loss)

        experiment = self.save(
            training_epochs=training_epochs,
            training_loss_components=training_mean_loss_component_collector,
        )

        return experiment, self._model

    def _generate_initial_state_files(self):

        if self._initial_state_generator is None:
            return []

        params = self._initial_state_generator.get_generator_params()

        levels = params['train']
        initial_states = self._initial_state_generator.generate_samples(1)
        initial_state_files = self._get_initial_states(initial_states, levels)

        return initial_state_files

    def _get_initial_states(self, batch, envs):

        initial_state_files = []
        for i, env in enumerate(envs):
            initial_state = np.array(np.swapaxes(batch[i]['input_observation_batch'][0], 2, 0) * 255.0).astype(np.uint8)
            state_filename = f'{env}_initial.npy'
            np.save(state_filename, initial_state)

            initial_state_files.append({
                'filename': state_filename,
                'caption': f'Initial state for training level: {env}'
            })

        return initial_state_files

    def save(self, training_epochs, training_loss_components):

        filename = 'model.tch'

        torch.save(self._model.saveable(), open(filename, 'wb'))

        training_history_csv = self._create_training_history_csv('training_history.csv',
                                                                 training_loss_components.get_history())

        train_final_values = {f'train_{k}_final': f'{v:.8f}' for k, v in
                              training_loss_components.get_window_mean().items()}

        meta = {
            'epochs': training_epochs,

            **self._hyperparameters,

            'data_generator': self._data_generator.get_name(),
            'action_map': self._data_generator.get_action_mapping(),
            **self._data_generator.get_generator_params(),

            **train_final_values,
        }

        files = [
            {
                'filename': training_history_csv,
                'caption': 'Training history'
            },
            {
                'filename': filename,
                'caption': f'{self.get_name()}-{self._data_generator.get_name()}-model'
            }
        ]

        files.extend(self._generate_initial_state_files())

        return self._mr.save(f'{self.get_name()}', meta, files=files)

    def _create_training_history_csv(self, filename, history_data):

        dataframe = pd.DataFrame(history_data)

        dataframe.to_csv(filename, header=True)

        return filename
