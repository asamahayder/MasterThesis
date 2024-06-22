import numpy as np
import math
from scipy.signal.windows import tukey
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

def preprocess_data(data, params):
    print("Running Preprocessing")

    # Fixing labeling errors
    data[51]['samplematrix'] = 'sample 7 PBS'
    data[51]['conc'] = 0.0

    data[52]['samplematrix'] = 'sample 8 g/PBS'
    data[52]['conc'] = 2.5

    data[53]['samplematrix'] = 'sample 6 g/PBS'
    data[53]['conc'] = 2.5


    # Fixing issue with sample ids due to multiple days
    for d in data:
        d['samplematrix_fixed'] = d['samplematrix']

    for d in data[37:]:
        values = d['samplematrix'].split()
        if len(values) > 1:
            id = int(values[1])
            new_id = id+18
            new_samplematrix = values[0] + " " + str(new_id) + " " + values[2]
            d['samplematrix_fixed'] = new_samplematrix


    # Removing air and NC as these are irrelevant for our purpose
    data = [d for d in data if not d['samplematrix_fixed'] == 'air']

    ids_to_remove = []

    for d in data:
        values = d['samplematrix_fixed'].split()
        if values[2] == 'NC':
            ids_to_remove.append(values[1])

    data = [d for d in data if not d['samplematrix_fixed'].split()[1] in ids_to_remove]


    # Applying Tukey window to all signals
    window_size = params["tukey_window_size"]
    alpha = params["tukey_alpha"]

    for d in data:
        for scan in d['scan']:
            y = scan['forward_scan']['signal']

            min_index = np.argmin(y)
            max_index = np.argmax(y)
            middle_index = math.floor((min_index+max_index)/2)

            window_start = middle_index - window_size
            window_end = middle_index + window_size

            window = tukey(window_end-window_start, alpha = alpha)
            windowed_signal = y[window_start: window_end] * window
            y = np.zeros(len(y), dtype=float)
            y[window_start: window_end] = windowed_signal

            scan['signal_tukey'] = y

        if d['ref'] is None:
            continue

        for scan in d['ref']:
            y = scan['forward_scan']['signal']

            min_index = np.argmin(y)
            max_index = np.argmax(y)
            middle_index = math.floor((min_index+max_index)/2)

            window_start = middle_index - window_size
            window_end = middle_index + window_size

            window = tukey(window_end-window_start, alpha = alpha)
            windowed_signal = y[window_start: window_end] * window
            y = np.zeros(len(y), dtype=float)
            y[window_start: window_end] = windowed_signal

            scan['signal_tukey'] = y


    # Plotting effect of tukey window
    """
    x = data[4]['scan'][0]['forward_scan']['time']
    y = data[4]['scan'][0]['signal_tukey']

    min_index = np.argmin(y)
    max_index = np.argmax(y)
    middle_index = math.floor((min_index + max_index) / 2)
    zoom_start = middle_index - 100
    zoom_end = middle_index + 100

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(x, y, label='Full Signal')
    plt.xlabel("Time(s)")
    plt.ylabel("Signal (nA)")
    plt.title("Full Pulse")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(x[zoom_start:zoom_end], y[zoom_start:zoom_end], label='Zoomed Signal')
    plt.xlabel("Time(s)")
    plt.ylabel("Signal (nA)")
    plt.title("Zoomed Pulse")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("temp_plots/signals_tukey_visualized.png")
    """

    # Applying FFT to all signals
    for d in data:
        for scan in d['scan']:
            x = scan['forward_scan']['time']
            y = scan['signal_tukey']

            fft_result = np.fft.fft(y)

            amplitudes = np.abs(fft_result)
                
            amplitudes = amplitudes / len(y)

            # FFT output is symmetric, so taking just the first half
            half_point = len(y) // 2
            frequencies = np.fft.fftfreq(len(y), d=x[1] - x[0])[:half_point]
            frequencies = frequencies * (10**(-12))
            amplitudes = amplitudes[:half_point]
            scan['amplitudes'] = amplitudes
            scan['frequencies'] = frequencies

        if d['ref'] is None:
            continue

        for scan in d['ref']:
            x = scan['forward_scan']['time']
            y = scan['signal_tukey']

            fft_result = np.fft.fft(y)

            # Extracting amplitudes (magnitudes of the complex numbers)
            amplitudes = np.abs(fft_result)
                
            # Normalizing the amplitudes by the number of samples to get the correct scale
            amplitudes = amplitudes / len(y)

            half_point = len(y) // 2
            frequencies = np.fft.fftfreq(len(y), d=x[1] - x[0])[:half_point]
            frequencies = frequencies * (10**(-12))
            amplitudes = amplitudes[:half_point]
            scan['amplitudes'] = amplitudes
            scan['frequencies'] = frequencies


    # Plotting FFT

    """
    xf_scan = data[1]['scan'][4]['frequencies']
    yf_scan = data[1]['scan'][4]['amplitudes']

    xf_ref = data[1]['ref'][4]['frequencies']
    yf_ref = data[1]['ref'][4]['amplitudes']

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(xf_scan, yf_scan, label='skin')
    ax.plot(xf_ref, yf_ref, label='air reference')
    ax.set_ylim(-0.01, 0.3)

    ax.set_xlabel('Frequency [THz]')
    ax.set_ylabel('Amplitude')
    ax.set_title('FFT of Signal')

    # [left, bottom, width, height] as fractions
    inset_ax = inset_axes(ax, width="150%", height="150%", loc='upper right',
                        bbox_to_anchor=(0.5, 0.4, 0.4, 0.4),
                        bbox_transform=ax.transAxes)

    inset_ax.plot(xf_scan, yf_scan, label='Signal')
    inset_ax.plot(xf_ref, yf_ref, label='Signal')

    # defining the zoom
    inset_ax.set_xlim(0, 2)  
    inset_ax.set_ylim(0, .25)  

    inset_ax.grid(True)

    inset_ax.tick_params(axis='both', which='major', labelsize=8)

    mark_inset(ax, inset_ax, loc1=1, loc2=4, fc="none", ec="0.5")

    ax.legend()

    plt.savefig("temp_plots/fft_visualized.png")
    """




    # Normalizing the amplitudes using it's air reference
    for d in data:
        for i in range(0, len(d['scan'])):
            scan_amplitudes = d['scan'][i]['amplitudes']
            ref_amplitudes = d['ref'][i]['amplitudes']
            normalized_amplitudes = scan_amplitudes / ref_amplitudes
            d['scan'][i]['normal'] = normalized_amplitudes   


    # Plotting Normalized Amplitudes

    """
    xf = data[1]['scan'][0]['frequencies']
    yf = data[1]['scan'][0]['normal']

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(xf, yf, label='Skin / reference')
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Amplitude")
    plt.title("Normalized Amplitudes")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(xf[0:4000], yf[0:4000], label='Skin / reference')
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Amplitude")
    plt.title("Normalized Amplitudes Zoomed in")
    plt.legend()
    plt.grid(True)

    plt.savefig("temp_plots/fft_norm.png")




    # Plotting the normalized bare pulse vs its corresponding normalized treated pulse
    xf_bare = data[1]['scan'][0]['frequencies']
    yf_bare = data[1]['scan'][0]['normal']

    xf_treated = data[7]['scan'][0]['frequencies']
    yf_treated = data[7]['scan'][0]['normal']

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(xf_bare, yf_bare, label='Normalized bare')
    plt.plot(xf_treated, yf_treated, label='Normalized treated')
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Amplitude")
    plt.title("Bare vs Treated Normalized Amplitudes")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(xf_bare[0:4000], yf_bare[0:4000], label='Normalized bare')
    plt.plot(xf_treated[0:4000], yf_treated[0:4000], label='Normalized treated')
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Amplitude")
    plt.title("Bare vs Treated Normalized Amplitudes Zoomed in")
    plt.legend()
    plt.grid(True)

    plt.savefig("temp_plots/fft_norm_bare_vs_treated.png")
    """


    # Creating sample ID as separate field in data dict
    for d in data:
        d['sample_id'] = d['samplematrix_fixed'].split()[1]


    # Normalizing treated pulse using its corresponding bare pulse  (Is this the correct way?)
    bare_data = [d for d in data if 'bare' in d['samplematrix_fixed']]
    treated_data = [d for d in data if not 'bare' in d['samplematrix_fixed']]

    for i, d in enumerate(treated_data):
        treated = treated_data[i]
        bare = bare_data[i]

        for j in range(0, len(d['scan'])):
            normal_treated = treated['scan'][j]['normal']
            normal_bare = bare['scan'][j]['normal']

            normalized_final = normal_treated / normal_bare

            d['scan'][j]['normal_final'] = normalized_final

    data = treated_data

    i = 0
    j = 0

    """
    treated_datapoint = treated_data[i]
    corresponding_bare = bare_data[i]

    xf_bare = corresponding_bare['scan'][j]['frequencies']
    yf_bare = corresponding_bare['scan'][j]['normal']

    xf_treated = treated_datapoint['scan'][j]['frequencies']
    yf_treated = treated_datapoint['scan'][j]['normal']

    xf_final = treated_datapoint['scan'][j]['frequencies']
    yf_final = treated_datapoint['scan'][j]['normal_final']

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(xf_bare, yf_bare, label='Normalized bare')
    plt.plot(xf_treated, yf_treated, label='Normalized treated')
    plt.plot(xf_final, yf_final, label='Normalized final')
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Amplitude")
    plt.title("Bare vs Treated Normalized Amplitudes")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(xf_bare[0:4000], yf_bare[0:4000], label='Normalized bare')
    plt.plot(xf_treated[0:4000], yf_treated[0:4000], label='Normalized treated')
    plt.plot(xf_final[0:4000], yf_final[0:4000], label='Normalized final')
    plt.xlabel("Frequency (THz)")
    plt.ylabel("Amplitude")
    plt.title("Bare vs Treated Normalized Amplitudes Zoomed in")
    plt.legend()
    plt.grid(True)

    plt.savefig("temp_plots/fft_norm_final.png")
    """


    # Preparing data for PCA and further analysis:
    X = []
    y = []
    y_day = []

    for d in data:
        for scan in d['scan']:
            X.append(scan['normal_final'][0:3000]) # Using only the first 3000 entries as it becomes too noisy after. (see above for explanation)
            y.append(d['samplematrix_fixed'].split()[2])
            y_day.append(d['date'].day)

    X = np.array(X)

    
    return X, y
