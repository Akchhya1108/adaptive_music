from midi_loader import load_midi_files

# Load all MIDI files from data folder
midi_data = load_midi_files("./data")


print(f"Loaded {len(midi_data)} MIDI files.")

# Optional: show details of first one
if midi_data:
    print("First file note count:", len(midi_data[0]))
    print("Sample notes (first 10):", midi_data[0][:10])
