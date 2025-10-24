import numpy as np

def create_sequences(all_sequences, sequence_length=32):
    """
    Convert sequences of (pitch, start_time, duration) into input-target pairs
    for autoregressive training.
    """
    X = []
    y = []

    for seq in all_sequences:
        if len(seq) <= sequence_length:
            continue
        for i in range(len(seq) - sequence_length):
            X.append(seq[i:i+sequence_length])
            y.append(seq[i+sequence_length])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    return X, y

# Optional: test if run directly
if __name__ == "__main__":
    from midi_loader import load_midi_files
    from midi_to_tokens import midi_to_sequence

    midi_data_list = load_midi_files("../data")
    sequences = midi_to_sequence(midi_data_list)
    X, y = create_sequences(sequences, sequence_length=32)
    print("X shape:", X.shape)
    print("y shape:", y.shape)
