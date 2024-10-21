# Full code for the task

import numpy as np
import matplotlib.pyplot as plt
import scipy.io

# Load the datasets
file_A_path = 'dataset_A.mat'
file_H_path = 'dataset_H.mat'

# Load the .mat files
data_A = scipy.io.loadmat(file_A_path)
data_H = scipy.io.loadmat(file_H_path)

# Extract the datasets from the loaded files
dataset_A = data_A['dataset_A']
dataset_H = data_H['dataset_H']

# Extract each stimulus type (columns) for both datasets
face_stimulus_A = dataset_A[0, 0]
text_stimulus_A = dataset_A[0, 1]
speech_stimulus_A = dataset_A[0, 2]

face_stimulus_H = dataset_H[0, 0]
text_stimulus_H = dataset_H[0, 1]
speech_stimulus_H = dataset_H[0, 2]

# Flatten the trial data if it's nested in arrays
def process_timestamps(trial_data):
    """
    Flatten the nested structure of the trial data if necessary
    """
    flattened_data = []
    for trial in trial_data:
        if isinstance(trial, np.ndarray) and len(trial) > 0 and isinstance(trial[0], np.ndarray):
            flattened_data.append(trial[0].flatten())  # Flatten nested arrays
        else:
            flattened_data.append(trial)
    return flattened_data

# Flatten the trial data for both datasets
face_stimulus_A_flat = process_timestamps(face_stimulus_A)
text_stimulus_A_flat = process_timestamps(text_stimulus_A)
speech_stimulus_A_flat = process_timestamps(speech_stimulus_A)

face_stimulus_H_flat = process_timestamps(face_stimulus_H)
text_stimulus_H_flat = process_timestamps(text_stimulus_H)
speech_stimulus_H_flat = process_timestamps(speech_stimulus_H)

# Define bin width and time range
bin_width = 200
time_range = np.arange(0, 1000 + bin_width, bin_width)

def compute_firing_rate(timestamps_trials, time_range):
    """
    Compute the firing rate for each trial based on the time bins
    """
    firing_rates = []
    for trial in timestamps_trials:
        trial_firing_rate, _ = np.histogram(trial, bins=time_range)
        firing_rates.append(trial_firing_rate)
    
    # Compute average firing rate across trials
    avg_firing_rate = np.mean(firing_rates, axis=0)
    return avg_firing_rate

# Compute firing rates for each stimulus for dataset_A (region A neuron)
face_firing_A = compute_firing_rate(face_stimulus_A_flat, time_range)
text_firing_A = compute_firing_rate(text_stimulus_A_flat, time_range)
speech_firing_A = compute_firing_rate(speech_stimulus_A_flat, time_range)

# Compute firing rates for each stimulus for dataset_H (region H neuron)
face_firing_H = compute_firing_rate(face_stimulus_H_flat, time_range)
text_firing_H = compute_firing_rate(text_stimulus_H_flat, time_range)
speech_firing_H = compute_firing_rate(speech_stimulus_H_flat, time_range)

# Calculate total firing rates for each stimulus
total_firing_A = [np.sum(face_firing_A), np.sum(text_firing_A), np.sum(speech_firing_A)]
total_firing_H = [np.sum(face_firing_H), np.sum(text_firing_H), np.sum(speech_firing_H)]

# Stimulus labels
stimulus_labels = ['Face', 'Text', 'Speech']

# Preferred stimulus and order of preference for each neuron
preferred_stimulus_A = stimulus_labels[np.argmax(total_firing_A)]
preferred_stimulus_H = stimulus_labels[np.argmax(total_firing_H)]

stimulus_order_A = [stimulus_labels[i] for i in np.argsort(total_firing_A)[::-1]]
stimulus_order_H = [stimulus_labels[i] for i in np.argsort(total_firing_H)[::-1]]

# Plotting the results
plt.figure(figsize=(12, 8))

# Plot for neuron A (region A)
plt.subplot(1, 2, 1)
plt.plot(time_range[:-1], face_firing_A, label='Face', color='red')
plt.plot(time_range[:-1], text_firing_A, label='Text', color='green')
plt.axvline(x=0, color='green', linestyle='--', linewidth=1.5, label='Stimulus Onset')     
plt.axvline(x=1000, color='red', linestyle='--', linewidth=1.5, label='Stimulus Offset')  
plt.plot(time_range[:-1], speech_firing_A, label='Speech', color='blue')
plt.title(f'Neuron A (Region A): Preferred Stimulus = {preferred_stimulus_A}\nOrder of Preference: {stimulus_order_A}')
plt.xlabel('Time (ms)')
plt.ylabel('Average Firing Rate')
plt.legend()

# Plot for neuron H (region H)
plt.subplot(1, 2, 2)
plt.plot(time_range[:-1], face_firing_H, label='Face', color='red')
plt.plot(time_range[:-1], text_firing_H, label='Text', color='green')
plt.axvline(x=0, color='green', linestyle='--', linewidth=1.5, label='Stimulus Onset')     # Stimulus onset
plt.axvline(x=1000, color='red', linestyle='--', linewidth=1.5, label='Stimulus Offset')  # Stimulus offset
plt.plot(time_range[:-1], speech_firing_H, label='Speech', color='blue')
plt.title(f'Neuron H (Region H): Preferred Stimulus = {preferred_stimulus_H}\nOrder of Preference: {stimulus_order_H}')
plt.xlabel('Time (ms)')
plt.ylabel('Average Firing Rate')
plt.legend()
plt.suptitle("Firing Rates of Neurons in Region A and Region H for Different Stimuli", fontsize=16)
plt.tight_layout()
plt.show()
