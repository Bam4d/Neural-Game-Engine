import argparse
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cycler
from matplotlib.ticker import FuncFormatter
from moderage import ModeRage

if __name__ == "__main__":
    mr = ModeRage()

    parser = argparse.ArgumentParser(description='Plot training results')
    parser.add_argument('-i', '--experiment-id', help='The ID of the trained experiment')

    args = parser.parse_args()

    experiment_id = args.experiment_id

    experiment = mr.load(experiment_id, 'NGE Learner')

    fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(12,8))
    files = experiment.files

    measures_to_plot = defaultdict(lambda: {})

    filename_regex = '(?P<name>gvgai-[a-z0-9]+-lvl\d+-v0)_(?P<measure>.*)?\.csv'

    for file in files:
        filename = file['filename']
        filename_parts = re.search(filename_regex, filename)

        if filename_parts is not None:
            measures_to_plot[filename_parts.group('measure')][filename_parts.group('name')] = filename

    # Tile error data
    validation_levels = measures_to_plot['tile_error_data']

    accuracy_axes = ax[0]
    training_axes = ax[1]

    accuracy_axes.set_title('Observation Training')
    accuracy_axes.set_prop_cycle(
        cycler('color', ['r', 'm', 'b', 'y', 'c']) + cycler('linestyle', ['-', '--', ':', '-.', '-']))

    for label in sorted(validation_levels):
        filename = validation_levels[label]
        data_frame = pd.read_csv(experiment.get_file(filename))
        data_frame.plot(x='epoch', y='acc', ax=accuracy_axes, label=label)

    accuracy_axes.set_xlim([0, 5000])
    accuracy_axes.set_ylim(bottom=0.9, top=1.01)
    accuracy_axes.set_xlabel(None)

    accuracy_axes.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x / 1000:g}K'))
    accuracy_axes.set(ylabel='Accuracy $(E_{t})$')
    accuracy_axes.get_legend().remove()

    validation_levels = measures_to_plot['mean_squared_error']

    training_axes.set_prop_cycle(
        cycler('color', ['r', 'm', 'b', 'y', 'c']) + cycler('linestyle', ['-', '--', ':', '-.', '-']))

    for label in sorted(validation_levels):
        filename = validation_levels[label]
        data_frame = pd.read_csv(experiment.get_file(filename))
        data_frame.plot(x='epoch', y='mean', ax=training_axes, label=label)

    training_axes.set_xlim([0, 5000])
    training_axes.set_yscale('log')
    training_axes.set_ylim(bottom=10e-7, top=10e-2)
    training_axes.set_xlabel(None)
    training_axes.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x / 1000:g}K'))

    training_axes.set(ylabel='Error $(E_{mse})$')

    fig.tight_layout()
    fig.subplots_adjust(top=0.9)

    fig.savefig("training_.pdf")
