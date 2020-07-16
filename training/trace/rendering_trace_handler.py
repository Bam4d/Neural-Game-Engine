from torch import nn
import logging

import numpy as np


class TraceHandler():

    def __init__(self, name):
        self._name = name

    def trace(self, step, total_steps, decoded_state, internal_state):
        raise NotImplementedError


class RenderingTraceHandler(TraceHandler):

    def __init__(self, name, width, height, headless=False):
        super().__init__(name)

        self._width = width
        self._height = height

        self._headless = headless

        self._pyglet = __import__('pyglet')
        self._gl = self._pyglet.gl

        if not headless:
            self.window = self._pyglet.window.Window(width=width, height=height, vsync=False, resizable=True)


class GymDataRenderer(RenderingTraceHandler):

    def __init__(self, width, height, steps=1, summary_writer=None, headless=False):
        super().__init__("Gym Renderer", width, height * steps, headless)

        self._summary_writer = summary_writer
        self._trace_counter = 0

    def trace(self, step, total_steps, decoded_state, internal_state):
        # Just run the trace on the first item in the batch
        state_trace = decoded_state.detach().cpu().numpy()[0]

        if not hasattr(self, '_buffer'):
            self._buffer = np.zeros((total_steps, state_trace.shape[2], state_trace.shape[1], 3), dtype=np.uint8)

        self._buffer[step] = np.round(np.swapaxes(state_trace, 0, 2) * 255)

        if self._summary_writer is not None:
            self._summary_writer.add_image(f'trace-{step}', self._buffer[step], self._trace_counter, dataformats='HWC')

        if not self._headless and step == (total_steps - 1):
            image = self.get_image(self._buffer, total_steps)

            texture = image.get_texture()
            texture.width = self._width
            texture.height = self._height
            self.window.clear()
            self.window.switch_to()
            self.window.dispatch_events()
            self._gl.glTexParameteri(self._gl.GL_TEXTURE_2D, self._gl.GL_TEXTURE_MIN_FILTER, self._gl.GL_NEAREST)
            self._gl.glTexParameteri(self._gl.GL_TEXTURE_2D, self._gl.GL_TEXTURE_MAG_FILTER, self._gl.GL_NEAREST)
            texture.blit(0, 0)  # draw
            self.window.flip()

            self._trace_counter += 1

    def get_image(self, buffer, steps=1):
        return self._pyglet.image.ImageData(buffer.shape[2],
                                            buffer.shape[1] * steps,
                                            'RGB',
                                            buffer.tobytes(),
                                            pitch=buffer.shape[2] * -3
                                            )
