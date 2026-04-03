import os
import torch

# Directory configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data", "MIDI")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

# Ensure output directories exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# Hyperparameters
SEQ_LENGTH = 50 #number of previous notes to consider for predicting the next note
BATCH_SIZE = 64
EPOCHS = 50
LEARNING_RATE = 0.001
EMBED_SIZE = 256
HIDDEN_SIZE = 512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")