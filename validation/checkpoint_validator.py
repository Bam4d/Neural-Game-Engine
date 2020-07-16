import logging
from collections import defaultdict
import numpy as np

from griddly.RenderTools import VideoRecorder

class CallBackValidator():

    def __init__(self, validation_repeats, validation_steps, ngpu_iterations, level_generator, wandb_logger=None):

        self._history = defaultdict(lambda: defaultdict(lambda: []))

        self._repeats = validation_repeats
        self._steps = validation_steps
        self._ngpu_iterations = ngpu_iterations

        self._level_generator = level_generator

        self._wandb_logger = wandb_logger

    def get_callback(self, model):
        logger = logging.getLogger("Generalisation")

        self.compare_levels = [level for level in range(self._level_generator.get_num_levels())]

        def checkpoint_callback(e):
            # Then print a trace of learning a value
            for i, batch in enumerate(self._level_generator.generate_samples(32)):
                video_shape = np.array(batch['input_observation_batch'][0].shape) * [1, 3, 1]
                video_recorder = VideoRecorder()
                video_filename = f'level_{i}_{e}.mp4'
                video_recorder.start(video_filename, video_shape)
                t_batch = model._model.prepare_batch(batch)
                # Only view out the largest
                trace = i == len(self.compare_levels) - 1
                level = self.compare_levels[i]

                (loss, _), predicted = model.eval(t_batch, trace=trace)
                logger.info(f'Loss at {level}: {loss:.4f}')

                for f in range(32):
                    input_frame = batch['input_observation_batch'][f]
                    expected_frame = batch['expected_observation_batch'][f]
                    predicted_frame = predicted['observation_predictions'][f]
                    video_frame = np.concatenate([input_frame, expected_frame, predicted_frame.detach().cpu().numpy()], axis=1)
                    video_frame = (video_frame * 255).astype('uint8')
                    video_recorder.add_frame(video_frame)

                #self._wandb_logger.log({'frame_compare': wandb.Video(video_filename)}, commit=false)
                #self._wandb_logger.log({f'level {level} loss': loss}, commit=False)

        return checkpoint_callback
