import torch
import numpy as np
from music21 import note, chord, instrument, stream
import os
import argparse
from config import *
from utils import load_pickle
from train import MusicLSTM

def generate_music(mode, num_notes=200):
    print(f"--- Generating {mode.upper()} Music ---")
    
    # Load mappings and model
    note_to_int = load_pickle(os.path.join(PROCESSED_DIR, f"{mode}_mapping.pkl"))
    vocab_size = load_pickle(os.path.join(PROCESSED_DIR, f"{mode}_vocab.pkl"))
    int_to_note = {number: note for note, number in note_to_int.items()}
    X = torch.load(os.path.join(PROCESSED_DIR, f"{mode}_X.pt"))

    model = MusicLSTM(vocab_size, EMBED_SIZE, HIDDEN_SIZE).to(DEVICE)
    try:
        model.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"lstm_{mode}.pth")))
    except FileNotFoundError:
        print("Trained model not found. Run train.py first.")
        return
    
    model.eval()

    # Pick a random starting sequence from our training data
    start = np.random.randint(0, len(X)-1)
    pattern = X[start].tolist()
    prediction_output = []

    # Generate notes
    print("Generating notes...")
    for note_index in range(num_notes):
        prediction_input = torch.tensor([pattern], dtype=torch.long).to(DEVICE)
        
        with torch.no_grad():
            prediction = model(prediction_input)
            # Add some randomness (temperature) if needed, but argmax is safe
            index = torch.argmax(prediction, dim=1).item()
            
        result = int_to_note[index]
        prediction_output.append(result)
        
        pattern.append(index)
        pattern = pattern[1:] # slide window forward

    # Convert numeric output back to music21 objects
    offset = 0
    output_notes = []
    for pattern_note in prediction_output:
        # Pattern is a chord
        if ('.' in pattern_note) or pattern_note.isdigit():
            notes_in_chord = pattern_note.split('.')
            notes = []
            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        # Pattern is a note
        else:
            new_note = note.Note(pattern_note)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)
        # Increase offset so notes don't stack completely
        offset += 0.5 

    # Save to MIDI
    midi_stream = stream.Stream(output_notes)
    output_path = os.path.join(OUTPUTS_DIR, f"generated_{mode}.mid")
    midi_stream.write('midi', fp=output_path)
    print(f"Music generated and saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['hiphop', 'retro', 'mixed'])
    args = parser.parse_args()
    generate_music(args.mode)