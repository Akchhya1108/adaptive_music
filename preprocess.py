from music21 import converter, note
import os, json
from tqdm import tqdm

DATA_DIR = "data/midi"
OUT_FILE = "data/notes.json"

def midi_to_notes(path):
    midi = converter.parse(path)
    notes = []
    for element in midi.flat.notes:
        if isinstance(element, note.Note):
            notes.append(element.pitch.midi)
    return notes

def main():
    all_notes = []
    for f in tqdm(os.listdir(DATA_DIR)):
        if f.endswith(".mid") or f.endswith(".midi"):
            path = os.path.join(DATA_DIR, f)
            notes = midi_to_notes(path)
            all_notes.extend(notes)
    json.dump(all_notes, open(OUT_FILE, "w"))
    print("Saved notes to", OUT_FILE)

if __name__ == "__main__":
    main()
