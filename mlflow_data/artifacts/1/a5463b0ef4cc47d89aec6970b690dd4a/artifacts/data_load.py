import os
import pickle
from datetime import datetime

def load_data(_, path_to_data_folder):

    print(path_to_data_folder)
    print(os.getcwd())

    files = os.listdir(path_to_data_folder)

    files.remove("READ_ME.txt")

    data = []

    for file in files:
        with open(path_to_data_folder + file, 'rb') as f:
            data.append(pickle.load(f))
    
    for d in data:
        timestamp = d['time']
        d['date'] = datetime.fromisoformat(timestamp)

    data = sorted(data, key=lambda x: x['date'])

    return data