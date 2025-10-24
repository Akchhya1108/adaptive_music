import os
import pretty_midi

def load_midi_files(folder_path):
    midi_files = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".mid") or filename.endswith(".midi"):
            try:
                midi = pretty_midi.PrettyMIDI(os.path.join(folder_path, filename))
                midi_files.append(midi)
            except Exception as e:
                print(f"Failed to load {filename}: {e}")
    return midi_files
