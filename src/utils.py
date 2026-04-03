import pickle
import os

def save_pickle(obj, filepath):
    """Saves an object to a pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)

def load_pickle(filepath):
    """Loads an object from a pickle file."""
    if not os.path.exists(filepath):
        return None
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def get_midi_files(mode, data_dir):
    """Returns a list of MIDI file paths based on the selected mode."""
    files = []
    modes_to_load = []
    
    if mode == "hiphop":
        modes_to_load = ["HipHop"]
    elif mode == "retro":
        modes_to_load = ["RetroGame"]
    elif mode == "mixed":
        modes_to_load = ["HipHop", "RetroGame"]
    else:
        raise ValueError("Invalid mode. Choose 'hiphop', 'retro', or 'mixed'.")

    for folder in modes_to_load:
        folder_path = os.path.join(data_dir, folder)
        if os.path.exists(folder_path):
            for file in os.listdir(folder_path):
                if file.endswith(".mid") or file.endswith(".midi"):
                    files.append(os.path.join(folder_path, file))
    return files