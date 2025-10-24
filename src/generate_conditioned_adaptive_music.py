import torch
import numpy as np
from conditioned_lstm import ConditionedLSTM
from midi_loader import load_midi_files
from midi_to_tokens import midi_to_sequence
from pathlib import Path
from mido import Message, MidiFile, MidiTrack

# ---------------------------
# PARAMETERS
# ---------------------------
MODEL_PATH = "../models/conditioned_lstm.pth"
OUTPUT_MIDI = "../generated/adaptive_generated.mid"
SEQUENCE_LENGTH = 32
NUM_STEPS = 200
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# LOAD MODEL
# ---------------------------
model = ConditionedLSTM(input_size=3, cond_size=3, hidden_size=256, num_layers=2, output_size=3).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# ---------------------------
# HELPER: Convert sequence to MIDI
# ---------------------------
def sequence_to_midi(sequence, output_path):
    midi = MidiFile()
    track = MidiTrack()
    midi.tracks.append(track)

    for note, start, dur in sequence:
        note = int(round(note))
        start = float(start)
        dur = float(dur)
        track.append(Message('note_on', note=note, velocity=64, time=int(start*480)))
        track.append(Message('note_off', note=note, velocity=64, time=int(dur*480)))
    Path(output_path).parent.mkdir(exist_ok=True, parents=True)
    midi.save(output_path)
    print(f"MIDI saved to {output_path}")

# ---------------------------
# GENERATE ADAPTIVE MUSIC
# ---------------------------
def generate_conditioned_music(start_sequence, conditioning_sequence, num_steps=NUM_STEPS):
    """
    start_sequence: list of shape (seq_len, 3) containing initial MIDI events
    conditioning_sequence: list of shape (num_steps, cond_size) representing game-state
    """
    generated = start_sequence.copy()

    for t in range(num_steps):
        # take last SEQUENCE_LENGTH steps as input
        seq_input = np.array(generated[-SEQUENCE_LENGTH:], dtype=np.float32).reshape(1, -1, 3)
        cond_input = np.array(conditioning_sequence[t], dtype=np.float32).reshape(1, -1)

        seq_tensor = torch.tensor(seq_input, dtype=torch.float32).to(DEVICE)
        cond_tensor = torch.tensor(cond_input, dtype=torch.float32).to(DEVICE)

        with torch.no_grad():
            next_note = model(seq_tensor, cond_tensor)  # (1,3)
        generated.append(next_note.cpu().numpy().flatten())

    return np.array(generated)

# ---------------------------
# EXAMPLE USAGE
# ---------------------------
if __name__ == "__main__":
    # Load starting MIDI sequence
    midi_data = load_midi_files("../data")
    sequences = midi_to_sequence(midi_data)
    start_sequence = sequences[0][:SEQUENCE_LENGTH]  # take first seq

    # Create a dummy conditioning sequence for demo
    # Here cond = [combat_intensity, tempo_factor, tension_level] for example
    conditioning_sequence = np.zeros((NUM_STEPS, 3), dtype=np.float32)
    for i in range(NUM_STEPS):
        if i < NUM_STEPS//3:
            conditioning_sequence[i] = [1.0, 1.0, 0.2]  # calm
        elif i < 2*NUM_STEPS//3:
            conditioning_sequence[i] = [2.0, 1.2, 0.8]  # combat
        else:
            conditioning_sequence[i] = [1.0, 1.0, 0.1]  # exploration

    generated_seq = generate_conditioned_music(start_sequence, conditioning_sequence, num_steps=NUM_STEPS)
    sequence_to_midi(generated_seq, OUTPUT_MIDI)
