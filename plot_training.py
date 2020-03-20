import argparse
import re
import numpy as np
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cycler
from matplotlib.ticker import FuncFormatter
from moderage import ModeRage

def plot_results_for_experiment(ax, idx, experiment, color=None, facecolor=None, col_title=None, label=None):
    files = experiment.files

    measures_to_plot = defaultdict(lambda: {})

    tile_filename_regex = 'tile_(?P<id>.*)?_(?P<measure>.*)?\.csv'

    num_tiles = 0
    for file in files:
        filename = file['filename']
        filename_parts = re.search(tile_filename_regex, filename)

        if filename_parts is not None:

            tile_id = int(filename_parts.group('id'))

            if tile_id > num_tiles:
                num_tiles = tile_id

            measures_to_plot[filename_parts.group('measure')][tile_id] = filename

    # Tile f1 score
    tile_f1_scores = measures_to_plot['f1']

    all_tile_f1 = []
    for t in range(num_tiles):
        padded = np.full((25), np.nan)
        tile_f1 = pd.read_csv(experiment.get_file(tile_f1_scores[t]))['mean'].to_numpy()

        padded[:tile_f1.shape[0]] = tile_f1

        all_tile_f1.append(padded)

    tile_f1_data = np.stack(all_tile_f1)

    tile_f1_mean = np.nanmean(tile_f1_data, axis=0)
    tile_f1_max = np.nanmax(tile_f1_data, axis=0)
    tile_f1_min = np.nanmin(tile_f1_data, axis=0)

    x = np.arange(25) * 200 * 8 * 32

    if len(ax.shape) == 1:
        accuracy_axes = ax[0]
        training_axes = ax[1]
    else:
        accuracy_axes = ax[0, idx]
        training_axes = ax[1, idx]

    if col_title is not None:
        accuracy_axes.set_title(col_title)

    mean_mask = np.isfinite(tile_f1_mean)
    fill_mask = np.isfinite(tile_f1_max) * np.isfinite(tile_f1_min)

    accuracy_axes.plot(x[mean_mask], tile_f1_mean[mean_mask], color=color, label=label)
    accuracy_axes.fill_between(x[fill_mask], tile_f1_max[fill_mask], tile_f1_min[fill_mask], facecolor=facecolor, alpha=0.1)

    accuracy_axes.set_xlim([0, 1500000])
    accuracy_axes.set_xticks([])
    accuracy_axes.set_xlabel(None)

    if idx == 0:
        accuracy_axes.set(ylabel='Average Tile F-Score $(F_{t})$')
    else:
        accuracy_axes.set_yticks([])

    observation_mse = pd.read_csv(experiment.get_file('observation_mean_squared_error.csv'))['mean'].to_numpy()

    training_axes.plot(x, observation_mse, label=label)

    training_axes.set_xlim([0, 1500000])
    training_axes.set_xticks([0, 500000, 1000000, 1500000])
    training_axes.set_yscale('log')
    training_axes.set_ylim(bottom=10e-7, top=10e-2)
    training_axes.set_xlabel(None)
    training_axes.xaxis.set_major_formatter(FuncFormatter(lambda x, p: f'{x / 1000000:g}'))

    if idx == 0:
        training_axes.set(ylabel='Error $(E_{mse})$')
    else:
        training_axes.set_yticks([])

    print(f'F_t: {tile_f1_min[-1]} E_mse: {observation_mse[-1]}')

    return training_axes.get_legend_handles_labels()

if __name__ == "__main__":
    mr = ModeRage()

    parser = argparse.ArgumentParser(description='Plot training results')
    parser.add_argument('-i', '--experiment-id', help='The ID of the trained experiment')

    args = parser.parse_args()

    experiment_id = args.experiment_id

    experiment = mr.load(experiment_id, 'NGE_Learner')

    fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(12,8))

    plot_results_for_experiment(ax, 0, experiment)

    fig.savefig("train.pdf", bbox_inches='tight', color='m', facecolor='b', pad_inches=0)

