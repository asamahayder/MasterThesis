import os
import pickle
from datetime import datetime
from tqdm import tqdm

def load_data(_, params):
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
    
    return data
