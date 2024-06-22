import os
import pickle
from datetime import datetime
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import math

def load_data(_, params, mlflow = None):
    print("Running Data Load")

    final_data_path = os.path.join(params["path_to_data_folder"], "final_data.pkl")
    
    # Check if the final data pickle file exists
    if os.path.exists(final_data_path):
        # Load the final data object from the pickle file
        with open(final_data_path, 'rb') as f:
            data = pickle.load(f)
        print("Loaded final data from pickle file.")
    else:
        # Load individual files and create the final data object
        files = os.listdir(params["path_to_data_folder"])
        
        if "READ_ME.txt" in files:
            files.remove("READ_ME.txt")

        data = []

        for file in tqdm(files, desc="Loading files"):
            with open(os.path.join(params["path_to_data_folder"], file), 'rb') as f:
                data.append(pickle.load(f))
        
        for d in data:
            timestamp = d['time']
            d['date'] = datetime.fromisoformat(timestamp)

        data = sorted(data, key=lambda x: x['date'])
        
        # Save the final data object as a pickle file
        with open(final_data_path, 'wb') as f:
            pickle.dump(data, f)
        print("Final data saved to pickle file.")


    # Plot signals
    x = data[4]['scan'][0]['forward_scan']['time']
    y = data[4]['scan'][0]['forward_scan']['signal']

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
    plt.show()

    
    # Log the plot as an artifact
    if mlflow:
        # Save the plot to a file
        plot_path = "signals_visualized.png"
        plt.savefig(plot_path)
        mlflow.log_artifact(plot_path)

        # Remove the plot file after logging
        os.remove(plot_path)
        
    return data
