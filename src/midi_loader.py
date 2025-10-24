import pretty_midi
import numpy as np
import os

def load_midi_files(folder_path):
    midi_data = []
    for file in os.listdir(folder_path):
        if file.endswith(".mid") or file.endswith(".midi"):
            try:
                midi = pretty_midi.PrettyMIDI(os.path.join(folder_path, file))
                notes = []
                for instrument in midi.instruments:
                    for note in instrument.notes:
                        notes.append((note.pitch, note.start, note.end))
                midi_data.append(notes)
            except Exception as e:
                print(f"Error loading {file}: {e}")
    return midi_data
