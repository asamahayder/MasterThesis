import numpy as np
import math
from scipy.signal.windows import tukey

def preprocess_data(data, params, mlflow = None):
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


    # Normalizing the amplitudes using it's air reference
    for d in data:
        for i in range(0, len(d['scan'])):
            scan_amplitudes = d['scan'][i]['amplitudes']
            ref_amplitudes = d['ref'][i]['amplitudes']
            normalized_amplitudes = scan_amplitudes / ref_amplitudes
            d['scan'][i]['normal'] = normalized_amplitudes   


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
