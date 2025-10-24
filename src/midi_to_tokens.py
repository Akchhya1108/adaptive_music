import pretty_midi

def midi_to_sequence(midi_data_list):
    """
    Convert PrettyMIDI objects to sequences of (pitch, start, duration)
    Returns a list of sequences, one per MIDI file.
    """
    all_sequences = []

    for item in midi_data_list:
        # if item is a tuple, assume first element is PrettyMIDI object
        if isinstance(item, tuple):
            midi = item[0]
        else:
            midi = item

        sequence = []
        for instrument in midi.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    sequence.append((note.pitch, note.start, note.end - note.start))

        # Sort sequence by start time
        sequence = sorted(sequence, key=lambda x: x[1])
        if sequence:  # only append non-empty sequences
            all_sequences.append(sequence)

    return all_sequences


