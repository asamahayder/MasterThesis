import numpy as np
import math
from scipy.signal.windows import tukey
import matplotlib.pyplot as plt
from scipy.signal import correlate
import logger

def preprocess_data(data, params):
    # Choose the first pulse as the reference pulse (you can change this as needed)
    reference_pulse = data[0]['scan'][0]['forward_scan']['signal']
    min_index = np.argmin(reference_pulse)
    max_index = np.argmax(reference_pulse)
    middle_index = math.floor((min_index+max_index)/2)

    pulse_window_size = params['pulse_window_size']
    window_start = middle_index - pulse_window_size
    window_end = middle_index + pulse_window_size

    # Function to find the shift using cross-correlation
    def find_shift(reference, pulse):
        correlation = correlate(pulse, reference)
        shift = correlation.argmax() - (len(pulse) - 1)
        return shift

    # Align the pulses
    aligned_data = []
    for pulse in data:
        for scan in pulse['scan']:
            signal = scan['forward_scan']['signal']
            shift = find_shift(reference_pulse, signal)
            aligned_pulse = np.roll(signal, -shift)
            aligned_pulse = aligned_pulse[window_start:window_end] # Crop the signal to the window so all pulses have the same length
            scan['aligned'] = aligned_pulse
            aligned_data.append(aligned_pulse)
        for ref in pulse['ref']:
            signal = ref['forward_scan']['signal']
            shift = find_shift(reference_pulse, signal)
            aligned_pulse = np.roll(signal, -shift)
            aligned_pulse = aligned_pulse[window_start:window_end] # Crop the signal to the window so all pulses have the same length
            ref['aligned'] = aligned_pulse
            aligned_data.append(aligned_pulse)
        
        pulse['avg_pulse_aligned'] = np.mean([scan['aligned'] for scan in pulse['scan']], axis=0)
        pulse['avg_ref_aligned'] = np.mean([ref['aligned'] for ref in pulse['ref']], axis=0)



    plt.figure(figsize=(15, 5))

    number_of_pulses = len(aligned_data)

    for i in range(0, number_of_pulses):
        x = np.arange(len(aligned_data[i]))
        y = aligned_data[i]

        min_index = np.argmin(y)
        max_index = np.argmax(y)
        middle_index = math.floor((min_index + max_index) / 2)
        zoom_start = middle_index - 50
        zoom_end = middle_index + 50

        color = plt.cm.coolwarm(i / number_of_pulses)

        plt.plot(x[zoom_start:zoom_end], y[zoom_start:zoom_end], label='Pulse ' + str(i), color=color)

    plt.xlabel("Time(s)")
    plt.ylabel("Signal (nA)")
    plt.title("Zoomed Pulses")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("temp_plots/signals_aligned_zoomed.png")

    # plotting without zoom

    plt.figure(figsize=(15, 5))

    number_of_pulses = len(aligned_data)

    for i in range(0, number_of_pulses):
        x = np.arange(len(aligned_data[i]))
        y = aligned_data[i]

        color = plt.cm.coolwarm(i / number_of_pulses)

        plt.plot(x, y, label='Pulse ' + str(i), color=color)

    plt.xlabel("Time(s)")
    plt.ylabel("Signal (nA)")
    plt.title("Pulses")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("temp_plots/signals_aligned.png")




    # Using the averaged reference pulse to remove baseline noise from the average pulse

    for pulse in data:
        pulse['avg_pulse_aligned_cleaned'] = pulse['avg_pulse_aligned'] - pulse['avg_ref_aligned']

    

    # plotting without zoom

    plt.figure(figsize=(15, 5))

    number_of_pulses = len(data)

    for i in range(0, number_of_pulses):
        x = np.arange(len(data[i]['avg_pulse_aligned_cleaned']))
        y = data[i]['avg_pulse_aligned_cleaned']

        color = plt.cm.coolwarm(i / number_of_pulses)

        plt.plot(x, y, label='Pulse ' + str(i), color=color)

    plt.xlabel("Time(s)")
    plt.ylabel("Signal (nA)")
    plt.title("Cleaned Pulses")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("temp_plots/signal_air_subtracted.png")


    # using the bare pulses to remove sample noise from the treated pulse and to isolate the effect of the treatment

    bare_data = [d for d in data if 'bare' in d['samplematrix_fixed']]
    treated_data = [d for d in data if not 'bare' in d['samplematrix_fixed']]
    final_data = []
    labels = []
    for i, d in enumerate(treated_data):
        treated = treated_data[i]
        bare = bare_data[i]

        final_data.append(treated['avg_pulse_aligned_cleaned'] - bare['avg_pulse_aligned_cleaned'])
        labels.append(treated['samplematrix_fixed'].split()[2])

    final_data = np.asarray(final_data)
    labels = np.asarray(labels)



    # plotting the final data

    plt.figure(figsize=(15, 5))

    number_of_pulses = len(final_data)

    for i in range(0, number_of_pulses):
        x = np.arange(len(final_data[i]))
        y = final_data[i]

        color = plt.cm.coolwarm(i / number_of_pulses)

        plt.plot(x, y, label='Pulse ' + str(i), color=color)

    plt.xlabel("Time(s)")
    plt.ylabel("Signal (nA)")
    plt.title("Final Data")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("temp_plots/signal_bare_subtracted.png")





    # applying tukey window to the final data

    tukey_window_size = params['tukey_window_size']
    tukey_alpha = params['tukey_alpha']

    final_tukey_data = []

    for pulse in final_data:
        window_start = len(pulse) // 2 - tukey_window_size
        window_end = len(pulse) // 2 + tukey_window_size

        window = tukey(window_end-window_start, alpha = tukey_alpha)
        tukey_window = pulse[window_start: window_end] * window

        new_pulse = np.zeros(len(pulse), dtype=float)
        new_pulse[window_start: window_end] = tukey_window
        final_tukey_data.append(new_pulse)

    final_tukey_data = np.asarray(final_tukey_data)



    # plotting the final data after applying the tukey window

    plt.figure(figsize=(15, 5))

    number_of_pulses = len(final_tukey_data)

    for i in range(0, number_of_pulses):
        x = np.arange(len(final_tukey_data[i]))
        y = final_tukey_data[i]

        color = plt.cm.coolwarm(i / number_of_pulses)

        plt.plot(x, y, label='Pulse ' + str(i), color=color)

    plt.xlabel("Time(s)")
    plt.ylabel("Signal (nA)")
    plt.title("Final Data with Tukey Window")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("temp_plots/signal_tukey_applied.png")

    logger.log("Shape of data after tukey applied: ", final_tukey_data.shape)


    X = final_tukey_data
    y = labels

    
    return X, y