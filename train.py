import argparse
import logging
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from training.environment.gvgai_data_generator import get_game_levels, GVGAIRandomGenerator, GVGAILevelDataGenerator
from training.nge_trainer import GymLearner
from validation.checkpoint_validator import CallBackValidator

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-G', '--game', required=True, help="The game to learn")
    parser.add_argument('-i', '--iterations', type=int, default=2, help='Number of NGPU steps per frame')
    parser.add_argument('-l', '--learning-rate', type=float, default=0.01, help='The learning rate')
    parser.add_argument('-e', '--epochs', default=5000, type=int, help='The number of epochs to train for')
    parser.add_argument('-S', '--validation-steps', type=int, default=100, help='The number of steps in validations')
    parser.add_argument('-R', '--validation-repeats', type=int, default=3, help='The number of repeated validations')

    args = parser.parse_args()

    ngpu_iterations = args.iterations
    learning_rate = args.learning_rate
    game_name = args.game
    epochs = args.epochs

    validation_repeats = args.validation_repeats
    validation_steps = args.validation_steps

    hyperparameters = {
        'state_channels': 64,
        'ngpu_iterations': ngpu_iterations,

        'batch_size': 32,
        'learning_rate': learning_rate,
        'gradient_clip': 0.1,

        'reward_loss_coeff': 0.02,
        'reward_state_channels': 128,
        'reward_class_weight': [0.5, 0.5],

        'saturation_limit': 0.99,
        'saturation_cost_weight': 0.001,

        'learning_rate_patience': 500,
        'learning_rate_decay_factor': 0.5,

        'observation_noise_std': 0.1,  # will be multiplied by 255

        'prediction_steps': 1,
        'prediction_step_increase_epoch': 500,

        'validation_repeats': validation_repeats,
        'validation_steps': validation_steps,

    }

    now = datetime.now()

    summary_writer = SummaryWriter(f'./tensorboard/data/{now.strftime("%Y%m%d-%H%M%S")}', flush_secs=5)

    original_levels = get_game_levels(game_name)

    data_generator = GVGAIRandomGenerator(game_name)
    initial_state_generator = GVGAILevelDataGenerator(original_levels)

    gym_learner = GymLearner(hyperparameters, data_generator, initial_state_generator=initial_state_generator,
                             summary_writer=summary_writer)

    # Create nge_gym environments for trained model
    ngpu_iterations = hyperparameters['ngpu_iterations']

    validation_steps = hyperparameters['validation_steps']
    validation_repeats = hyperparameters['validation_repeats']

    callback_validator = CallBackValidator(validation_repeats, validation_steps, ngpu_iterations, original_levels)

    experiment, _ = gym_learner.train(
        epochs,
        callback_epoch=200,
        checkpoint_callback=callback_validator.get_callback(gym_learner, summary_writer)
    )

    # Have to make sure GVGAI stops
    callback_validator.cleanup()
    data_generator.cleanup()
    initial_state_generator.cleanup()

    # Save the training and
    history_files = callback_validator.save_history()
    experiment.add_files([{'filename': f} for f in history_files])
