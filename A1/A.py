import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# Plotting function for raster plots with raw data
def plot_raster(ax, data, title, color='blue'):
    # Add spikes for each trial
    for i, trial in enumerate(data):
        trial_spikes = trial[0]  # Access the raw spike timings from the data
        ax.vlines(trial_spikes, i + 0.5, i + 1.5, color=color, label="Spikes" if i == 0 else "")  # Add label only once

    # Stimulus onset and offset markers
    ax.axvline(x=0, color='green', linestyle='--', label="Stimulus Onset")
    ax.axvline(x=1000, color='red', linestyle='--', label="Stimulus Offset")

    # Adding titles and labels
    ax.set_xlim([-1000, 2000])
    ax.set_title(title, fontsize=12)
    ax.set_ylabel('Trials')
    ax.set_xlabel('Time (ms)')
    ax.legend(loc='upper right')
    ax.grid(True, linestyle='--', alpha=0.5)

# Load the .mat files
data_H = scipy.io.loadmat('dataset_H.mat')
data_A = scipy.io.loadmat('dataset_A.mat')

# Extracting data from cell arrays
dataset_H = data_H['dataset_H']
dataset_A = data_A['dataset_A']

# Creating the figure with 6 subplots (3 rows x 2 columns)
fig, axes = plt.subplots(3, 2, figsize=(15, 10))

# Plotting for Face stimulus (first row)
plot_raster(axes[0, 0], dataset_H[0, 0], 'Region H - Face Stimulus')
plot_raster(axes[0, 1], dataset_A[0, 0], 'Region A - Face Stimulus')

# Plotting for Text stimulus (second row)
plot_raster(axes[1, 0], dataset_H[0, 1], 'Region H - Text Stimulus')
plot_raster(axes[1, 1], dataset_A[0, 1], 'Region A - Text Stimulus')

# Plotting for Speech stimulus (third row)
plot_raster(axes[2, 0], dataset_H[0, 2], 'Region H - Speech Stimulus')
plot_raster(axes[2, 1], dataset_A[0, 2], 'Region A - Speech Stimulus')

# Set overarching labels for the left and right columns
fig.suptitle('Neuronal Activity from Regions H and A for Different Stimuli', fontsize=16)
fig.text(0.5, 0.04, 'Time (ms)', ha='center', fontsize=14)
fig.text(0.04, 0.5, 'Trial', va='center', rotation='vertical', fontsize=14)

# Adjust layout for clarity
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
