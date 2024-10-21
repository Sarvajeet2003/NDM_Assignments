import scipy.io
import numpy as np
import matplotlib.pyplot as plt

# Load the data
dataset_H = scipy.io.loadmat('dataset_H.mat')['dataset_H']
dataset_A = scipy.io.loadmat('dataset_A.mat')['dataset_A']

# Function to extract spike times from the dataset
def extract_spike_times(dataset):
    stimuli_data = []
    for i in range(dataset.shape[1]):
        trials = dataset[0, i]  # Access the ith stimulus type
        trials_list = []
        for trial_idx in range(trials.size):
            spike_times = trials[trial_idx, 0].flatten()
            trials_list.append(spike_times)
        stimuli_data.append(trials_list)
    return stimuli_data

# Extract spike times for both datasets
spike_times_H = extract_spike_times(dataset_H)
spike_times_A = extract_spike_times(dataset_A)

# Define stimulus types
stimuli = ['Face', 'Text', 'Speech']

# Define time bins (from -1000 ms to 2000 ms in 50 ms bins)
time_bins = np.arange(-1000, 2001, 50)  # Bins edges
bin_centers = (time_bins[:-1] + time_bins[1:]) / 2  # For plotting

# Create a figure with 3 rows and 2 columns
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 8), sharex=True)

# Loop over each stimulus type
for i, stimulus in enumerate(stimuli):
    # Plot histogram for dataset_H (Region H)
    ax_H = axes[i, 0]
    all_spike_times_H = np.concatenate(spike_times_H[i])  # Combine all trials for Region H
    ax_H.hist(all_spike_times_H, bins=time_bins, color='b', alpha=0.7)  # Plot histogram
    ax_H.axvline(x=0, color='g', linestyle='-', linewidth=1.5, label='Stimulus Onset')  # Stimulus onset
    ax_H.axvline(x=1000, color='r', linestyle='-', linewidth=1.5, label='Stimulus Offset')  # Stimulus offset
    ax_H.set_ylabel('Spike Count')
    ax_H.set_xlim([-1000, 2000])
    ax_H.set_title(f'{stimulus} - Region H')
    ax_H.legend()

    # Plot histogram for dataset_A (Region A)
    ax_A = axes[i, 1]
    all_spike_times_A = np.concatenate(spike_times_A[i])  # Combine all trials for Region A
    ax_A.hist(all_spike_times_A, bins=time_bins, color='b', alpha=0.7)  # Plot histogram
    ax_A.axvline(x=0, color='g', linestyle='-', linewidth=1.5, label='Stimulus Onset')  # Stimulus onset
    ax_A.axvline(x=1000, color='r', linestyle='-', linewidth=1.5, label='Stimulus Offset')  # Stimulus offset
    ax_A.set_ylabel('Spike Count')
    ax_A.set_xlim([-1000, 2000])
    ax_A.set_title(f'{stimulus} - Region A')
    ax_A.legend()

# Set common labels for the x-axis
for ax in axes[-1, :]:
    ax.set_xlabel('Time (ms)')

# Add a main title for the entire figure
fig.suptitle('Peri Stimulus Time Histogram (PSTH) for Regions H and A Across Stimuli', fontsize=16)

plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to accommodate the main title
plt.show()
