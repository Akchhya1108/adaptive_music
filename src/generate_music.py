# src/generate_music.py
import os
import numpy as np
import torch
import pretty_midi
from model import MusicRNN

MODEL_PATH = "../models/music_rnn.pth"
OUTPUT_PATH = "../generated_music/output.mid"
SEQUENCE_LENGTH = 32  # must match training sequence_length

def sanitize_note_vec(v):
    """Return a flat [pitch, start, duration] list of floats."""
    # convert numpy arrays or nested lists to flat list of 3 floats
    arr = np.array(v, dtype=np.float32).flatten()
    if arr.size < 3:
        # pad with zeros if somehow shorter
        arr = np.pad(arr, (0, 3 - arr.size), mode="constant", constant_values=0.0)
    arr = arr[:3]  # trim if longer
    return [float(arr[0]), float(arr[1]), float(arr[2])]

def load_model(path=MODEL_PATH, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MusicRNN(input_size=3, hidden_size=128, num_layers=2).to(device)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device

def generate_music(start_sequence, num_steps=200, model_path=MODEL_PATH, seq_len=SEQUENCE_LENGTH):
    # Load model
    model, device = load_model(model_path)

    # Normalize/prepare start_sequence: ensure it's a list of [p, s, d]
    arr = np.array(start_sequence, dtype=np.float32)
    if arr.ndim == 1 and arr.size == 3:
        arr = arr.reshape(1, 3)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("start_sequence must be shape (N,3) or (3,)")

    generated = [sanitize_note_vec(x) for x in arr.tolist()]  # python list of lists

    # If seed shorter than seq_len, pad by repeating first note (or zeros)
    if len(generated) < seq_len:
        pad_needed = seq_len - len(generated)
        pad_item = generated[0] if len(generated) > 0 else [60.0, 0.0, 0.5]
        generated = [pad_item]*pad_needed + generated

    for step in range(num_steps):
        # Build input window: last seq_len items -> shape (1, seq_len, 3)
        window = np.array(generated[-seq_len:], dtype=np.float32)  # shape (seq_len,3)
        input_tensor = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)  # (1, seq_len, 3)

        with torch.no_grad():
            output = model(input_tensor)  # model returns shape (1, 3) per our model.py
        # Extract next_note vector
        next_vec = output[0].cpu().numpy()  # shape (3,)
        next_note = sanitize_note_vec(next_vec)  # ensure flat list of 3 floats

        # Optional: set start time for generated notes so timing is coherent.
        # If your model predicts absolute start times it's fine; if not, you may want to
        # convert duration-only or relative times. Here we trust model output.
        generated.append(next_note)

    return np.array(generated, dtype=np.float32)  # (N_generated, 3)

def save_as_midi(sequence, output_path=OUTPUT_PATH):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    midi = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=0)
    for note in sequence:
        if len(note) != 3:
            continue
        pitch, start, duration = note
        pitch = int(np.clip(round(pitch), 0, 127))
        start = float(start)
        duration = float(duration)
        end = start + max(0.01, duration)  # avoid zero/negative durations
        n = pretty_midi.Note(velocity=100, pitch=pitch, start=start, end=end)
        instrument.notes.append(n)
    midi.instruments.append(instrument)
    midi.write(output_path)
    print(f"✅ MIDI written to: {output_path}")

if __name__ == "__main__":
    # --- Option A: random seed (quick test) ---
    # seed_pitches roughly in piano range, random start times small, durations reasonable
    seed = np.column_stack([
        np.random.uniform(48, 72, SEQUENCE_LENGTH),    # pitch 48-72
        np.linspace(0, SEQUENCE_LENGTH*0.25, SEQUENCE_LENGTH),  # start times spaced
        np.random.uniform(0.05, 0.5, SEQUENCE_LENGTH)  # durations
    ])

    # --- Option B: Uncomment to use a real seed from your trained sequences ---
    # from midi_loader import load_midi_files
    # from midi_to_tokens import midi_to_sequence
    # midi_list = load_midi_files("../data")
    # sequences = midi_to_sequence(midi_list)
    # seed = np.array(sequences[0][:SEQUENCE_LENGTH], dtype=np.float32)  # first file, first 32 notes

    gen = generate_music(seed, num_steps=200)
    save_as_midi(gen, OUTPUT_PATH)

