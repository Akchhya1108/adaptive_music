import torch
import numpy as np
from music21 import stream, note, tempo
import json
from model import MusicLSTM 

# Load data
with open("data/notes.json", "r") as f:
    notes = json.load(f)

unique_notes = sorted(list(set(notes)))
note_to_int = {n: i for i, n in enumerate(unique_notes)}
int_to_note = {i: n for i, n in enumerate(unique_notes)}

# --- MOOD SETTINGS ---
mood_settings = {
    "calm": {"tempo": 60, "transpose": -2},
    "happy": {"tempo": 90, "transpose": 0},
    "tense": {"tempo": 120, "transpose": 2},
    "battle": {"tempo": 140, "transpose": 3},
    "exploration": {"tempo": 110, "transpose": 2.5},
    
}

# Choose your mood here 👇
MOOD = "exploration"  # change between calm, happy, tense, battle

# Load model
model = MusicLSTM(len(unique_notes))
checkpoint = torch.load("music_model.pt", map_location=torch.device("cpu"))
model.load_state_dict(checkpoint["model"])
vocab = checkpoint.get("vocab", None)
model.eval()

# Seed sequence
sequence_length = 100
seed = notes[:sequence_length]
generated = seed.copy()

for _ in range(200):  # generate 200 new notes
    seq_input = [note_to_int[n] for n in generated[-sequence_length:]]
    x = torch.tensor(seq_input).unsqueeze(0)
    with torch.no_grad():
        pred_output = model(x)

# if model returns (output, hidden)
if isinstance(pred_output, tuple):
    pred_output = pred_output[0]
    pred_note = int_to_note[int(torch.argmax(pred_output))]

    generated.append(pred_note)

# Convert to MIDI
output_stream = stream.Stream()
output_stream.append(tempo.MetronomeMark(number=mood_settings[MOOD]["tempo"]))

for n_str in generated:
    try:
        n = note.Note(n_str)
        n.transpose(mood_settings[MOOD]["transpose"], inPlace=True)
        n.quarterLength = 0.5
        output_stream.append(n)
    except:
        pass

# Save adaptive music
output_file = f"adaptive_{MOOD}.mid"
output_stream.write("midi", fp=output_file)

print(f"✅ Adaptive {MOOD} soundtrack generated and saved as {output_file}")
