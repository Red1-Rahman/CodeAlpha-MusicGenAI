import glob
import os
import torch
from music21 import converter, instrument, note, chord
from config import *
from utils import save_pickle, get_midi_files

def parse_midi_files(mode):
    print(f"--- Preprocessing {mode.upper()} MIDI files ---")
    files = get_midi_files(mode, DATA_DIR)
    notes = []

    for file in files:
        print(f"Parsing {file}")
        try:
            midi = converter.parse(file)
            notes_to_parse = None
            
            # Group based on instruments
            parts = instrument.partitionByInstrument(midi)
            if parts: # If it has instrument parts
                notes_to_parse = parts.parts[0].recurse() 
            else: # If flat structure
                notes_to_parse = midi.flat.notes

            for element in notes_to_parse:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))
        except Exception as e:
            print(f"Could not parse {file}: {e}")

    # Create mappings from note to integer
    pitchnames = sorted(set(item for item in notes))
    vocab_size = len(pitchnames)
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # Create input sequences and the corresponding outputs
    for i in range(0, len(notes) - SEQ_LENGTH, 1):
        sequence_in = notes[i:i + SEQ_LENGTH]
        sequence_out = notes[i + SEQ_LENGTH]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    # Convert to PyTorch Tensors
    X = torch.tensor(network_input, dtype=torch.long)
    y = torch.tensor(network_output, dtype=torch.long)

    # Save processed data and mappings based on mode
    save_pickle(note_to_int, os.path.join(PROCESSED_DIR, f"{mode}_mapping.pkl"))
    torch.save(X, os.path.join(PROCESSED_DIR, f"{mode}_X.pt"))
    torch.save(y, os.path.join(PROCESSED_DIR, f"{mode}_y.pt"))
    save_pickle(vocab_size, os.path.join(PROCESSED_DIR, f"{mode}_vocab.pkl"))

    print(f"Preprocessing complete. Total unique notes/chords: {vocab_size}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True, choices=['hiphop', 'retro', 'mixed'])
    args = parser.parse_args()
    parse_midi_files(args.mode)