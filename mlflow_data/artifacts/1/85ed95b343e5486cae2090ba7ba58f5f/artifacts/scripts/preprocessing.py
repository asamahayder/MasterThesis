import numpy as np
import math
from scipy.signal.windows import tukey
import matplotlib.pyplot as plt
from scipy.signal import correlate
import logger
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

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
        shift = np.argmax(correlation) - len(pulse)
        return shift

    max_shift = 0
    aligned_data = []
    for pulse in tqdm(data, desc='Processing Pulses'):
        for scan in pulse['scan']:
            signal_forward = scan['forward_scan']['signal']
            signal_backward = scan['backward_scan']['signal']
            shift_forward = find_shift(reference_pulse, signal_forward)
            shift_backward = find_shift(reference_pulse, signal_backward)
            aligned_pulse_forward = np.roll(signal_forward, -shift_forward)
            aligned_pulse_backward = np.roll(signal_backward, -shift_backward)

            if abs(shift_forward) > max_shift:
                max_shift = abs(shift_forward)
            if abs(shift_backward) > max_shift:
                max_shift = abs(shift_backward)

            #aligned_pulse_forward = aligned_pulse_forward[window_start:window_end] # Cropping the signal to the window so all pulses have the same length
            #aligned_pulse_backward = aligned_pulse_backward[window_start:window_end] # Cropping the signal to the window so all pulses have the same length
            scan['forward_scan']['aligned'] = aligned_pulse_forward
            scan['backward_scan']['aligned'] = aligned_pulse_backward
            aligned_data.append(aligned_pulse_forward)
            aligned_data.append(aligned_pulse_backward)
        for ref in pulse['ref']:
            signal_forward = ref['forward_scan']['signal']
            signal_backward = ref['backward_scan']['signal']
            shift_forward = find_shift(reference_pulse, signal_forward)
            shift_backward = find_shift(reference_pulse, signal_backward)
            aligned_pulse_forward = np.roll(signal_forward, -shift_forward)
            aligned_pulse_backward = np.roll(signal_backward, -shift_backward)

            if abs(shift_forward) > max_shift:
                max_shift = abs(shift_forward)
            if abs(shift_backward) > max_shift:
                max_shift = abs(shift_backward)
            #aligned_pulse_forward = aligned_pulse_forward[window_start:window_end] # Cropping the signal to the window so all pulses have the same length
            #aligned_pulse_backward = aligned_pulse_backward[window_start:window_end] # Cropping the signal to the window so all pulses have the same length
            ref['forward_scan']['aligned'] = aligned_pulse_forward
            ref['backward_scan']['aligned'] = aligned_pulse_backward
            aligned_data.append(aligned_pulse_forward)
            aligned_data.append(aligned_pulse_backward)

        # Averaging the reference pulses to get a single reference pulse
        ref_forward_scans_aligned = [ref['forward_scan']['aligned'] for ref in pulse['ref']]
        ref_backward_scans_aligned = [ref['backward_scan']['aligned'] for ref in pulse['ref']]
        pulse['avg_ref_aligned'] = np.mean(ref_forward_scans_aligned + ref_backward_scans_aligned, axis=0)


    logger.log("Max shift: ", max_shift)
    logger.log("Shape of aligned data before cutting: ", np.array(aligned_data).shape)

    # using max shift to cut off the ends of the pulses
    for pulse in tqdm(data, desc='Cutting off ends of pulses'):
        for scan in pulse['scan']:
            # cutting of each end by max_shift
            scan['forward_scan']['aligned'] = scan['forward_scan']['aligned'][max_shift:-max_shift]
            scan['backward_scan']['aligned'] = scan['backward_scan']['aligned'][max_shift:-max_shift]
        
        pulse['avg_ref_aligned'] = pulse['avg_ref_aligned'][max_shift:-max_shift]

    # Also cut for aligned_data (which is just used temporarily for logging and plotting)
    aligned_data = [pulse[max_shift:-max_shift] for pulse in aligned_data]

    logger.log("Shape of aligned data after cutting: ", np.array(aligned_data).shape)
    

    plt.figure(figsize=(15, 5))

    # Pick random 30 pulses from aligned data to plot

    random_indices = np.random.choice(len(aligned_data), 30, replace=False)


    for i in random_indices:
        x = np.arange(len(aligned_data[i]))
        y = aligned_data[i]

        min_index = np.argmin(y)
        max_index = np.argmax(y)
        middle_index = math.floor((min_index + max_index) / 2)
        zoom_start = middle_index - 50
        zoom_end = middle_index + 50

        plt.plot(x[zoom_start:zoom_end], y[zoom_start:zoom_end], label='Pulse ' + str(i))

    plt.xlabel("Time(s)")
    plt.ylabel("Signal (nA)")
    plt.title("Zoomed Pulses")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("temp_plots/signals_aligned_zoomed.png")

    # plotting without zoom

    plt.figure(figsize=(15, 5))

    for i in random_indices:
        x = np.arange(len(aligned_data[i]))
        y = aligned_data[i]

        plt.plot(x, y, label='Pulse ' + str(i))

    plt.xlabel("Time(s)")
    plt.ylabel("Signal (nA)")
    plt.title("Pulses")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("temp_plots/signals_aligned.png")


    # Using the averaged reference pulse to remove baseline noise from the average pulse

    for pulse in tqdm(data, desc='Removing Baseline Noise using air reference'):
        for scan in pulse['scan']:
            signal_forward = scan['forward_scan']['aligned']
            signal_backward = scan['backward_scan']['aligned']
            
            scan['forward_scan']['cleaned'] = signal_forward - pulse['avg_ref_aligned']
            scan['backward_scan']['cleaned'] = signal_backward - pulse['avg_ref_aligned']


    # plotting without zoom

    plt.figure(figsize=(15, 5))

    i = 0

    # Pick random 10 pulses from cleaned data to plot

    random_indices = np.random.choice(len(data), 5, replace=False)

    for idx in random_indices:
        pulse = data[idx]
        for scan in pulse['scan']:
            x = np.arange(len(scan['forward_scan']['cleaned']))
            y = scan['forward_scan']['cleaned']
            plt.plot(x, y, label='Pulse ' + str(i))
            i += 1
            
            x = np.arange(len(scan['backward_scan']['cleaned']))
            y = scan['backward_scan']['cleaned']
            plt.plot(x, y, label='Pulse ' + str(i))
            i += 1


    plt.xlabel("Time(s)")
    plt.ylabel("Signal (nA)")
    plt.title("Cleaned Pulses")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("temp_plots/signal_air_subtracted.png")


    treated_data = [d for d in data if not 'bare' in d['samplematrix_fixed']]
    final_data = []
    labels = []
    reserved_data = []
    reserved_labels = []

    for i in tqdm(range(len(treated_data)), desc='Processing Treated Data'):
        treated = treated_data[i]

        all_treated_scans = [scan['forward_scan']['cleaned'] for scan in treated['scan']] + [scan['backward_scan']['cleaned'] for scan in treated['scan']]

        for j in range(len(all_treated_scans)):

            if i >= len(treated_data) - 4:
                reserved_data.append(all_treated_scans[j])
                reserved_labels.append(treated['samplematrix_fixed'].split()[2])
            else:
                final_data.append(all_treated_scans[j])
                labels.append(treated['samplematrix_fixed'].split()[2])

    final_data = np.asarray(final_data)
    reserved_data = np.asarray(reserved_data)
    labels = np.asarray(labels)
    reserved_labels = np.asarray(reserved_labels)






    # plotting the final data

    plt.figure(figsize=(15, 5))

    # Pick 30 random indicies from final data

    random_indices = np.random.choice(len(final_data), 30, replace=False)

    for i in random_indices:
        x = np.arange(len(final_data[i]))
        y = final_data[i]

        plt.plot(x, y, label='Pulse ' + str(i))

    plt.xlabel("Time(s)")
    plt.ylabel("Signal (nA)")
    plt.title("Final Data")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("temp_plots/signal_bare_subtracted.png")






    # Label encoding the labels to 0s and 1s
    le = LabelEncoder()
    le.classes_ = np.array(["g/PBS", "PBS"])
    labels = le.transform(labels)
    reserved_labels = le.transform(reserved_labels)

    X = final_data
    y = labels

    
    return X, y, reserved_data, reserved_labels