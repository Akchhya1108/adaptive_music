# 🎵 Adaptive Music Generator

An AI-powered music generation system that creates dynamic, mood-based MIDI soundtracks using LSTM neural networks. Perfect for game developers, content creators, and music enthusiasts!

## ✨ Features

- **AI-Powered Generation**: LSTM neural network trained on MIDI files
- **Mood-Based Adaptation**: Generate music for different moods (calm, happy, tense, battle)
- **Dynamic Tempo Control**: Adjustable BPM per mood
- **Transpose Support**: Key shifting for different emotional tones
- **MIDI Output**: Industry-standard MIDI file format

## 🎮 Mood Settings

| Mood | Tempo (BPM) | Transpose | Use Case |
|------|-------------|-----------|----------|
| **Calm** | 60 | -2 | Ambient, meditation, peaceful scenes |
| **Happy** | 90 | 0 | Upbeat content, celebrations |
| **Tense** | 120 | +2 | Suspense, drama, action build-up |
| **Battle** | 140 | +3 | Combat, intense action sequences |

## 🚀 Quick Start

### Prerequisites

```bash
python 3.8+
pip
```

### Installation

1. **Clone the repository**
```bash
git clone <your-repo-url>
cd adaptive-music-generator
```

2. **Install dependencies**
```bash
pip install torch music21 numpy tqdm
```

3. **Prepare MIDI files**
```bash
# Create data directory
mkdir -p data/midi

# Add your MIDI files to data/midi/
# (at least 10-20 files recommended)
```

### Usage

#### Step 1: Preprocess MIDI Files
```bash
python preprocess.py
```
This extracts musical notes from your MIDI files and saves them to `data/notes.json`.

#### Step 2: Train the Model
```bash
python train.py
```
- Trains for 20 epochs (customizable)
- Saves model to `music_model.pt`
- Progress bar shows training status

#### Step 3: Generate Music
```bash
python sample.py
```

**Customize the mood** by editing `sample.py`:
```python
# Line 24
MOOD = "happy"  # Options: calm, happy, tense, battle
```

Output: `adaptive_{MOOD}.mid`

## 📁 Project Structure

```
adaptive-music-generator/
├── data/
│   ├── midi/              # Your input MIDI files
│   └── notes.json         # Processed notes (auto-generated)
├── dataset.py             # Dataset loader
├── model.py               # LSTM model architecture
├── preprocess.py          # MIDI → notes converter
├── train.py               # Training script
├── sample.py              # Music generation script
├── music_model.pt         # Trained model (auto-generated)
└── adaptive_*.mid         # Generated music (auto-generated)
```

## 🎹 Model Architecture

- **Input**: Sequence of 50 MIDI notes
- **Embedding Layer**: 64 dimensions
- **LSTM**: 2 layers, 128 hidden units
- **Output**: Next note prediction (vocab size)

```python
MusicLSTM(
  vocab_size=<unique_notes>,
  emb=64,
  hidden=128,
  num_layers=2
)
```

## ⚙️ Configuration

### Training Parameters (train.py)
```python
SEQ_LEN = 50        # Sequence length
EPOCHS = 20         # Training epochs
BATCH_SIZE = 64     # Batch size
LEARNING_RATE = 1e-3
```

### Generation Parameters (sample.py)
```python
sequence_length = 100  # Seed sequence length
num_notes = 200        # Notes to generate
```

## 🎨 Customization

### Add New Moods
Edit `sample.py`:
```python
mood_settings = {
    "mysterious": {"tempo": 75, "transpose": -3},
    "epic": {"tempo": 110, "transpose": 4},
    # Add your custom moods here
}
```

### Adjust Note Duration
```python
n.quarterLength = 0.5  # Change from 0.5 to 1.0 for longer notes
```

## 📊 Training Tips

1. **More MIDI files = Better results**
   - Minimum: 10-20 files
   - Recommended: 50+ files
   - Similar genre/style works best

2. **Training Time**
   - CPU: ~10-15 minutes (20 epochs)
   - GPU: ~2-3 minutes (20 epochs)

3. **Improve Quality**
   - Increase epochs (e.g., 50)
   - Add more training data
   - Experiment with model size

## 🔧 Troubleshooting

### "No MIDI files found"
```bash
# Ensure MIDI files are in correct location
ls data/midi/*.mid
```

### "Model not found"
```bash
# Train the model first
python train.py
```

### Poor Generation Quality
- Train for more epochs
- Add more diverse MIDI training data
- Adjust sequence length in sample.py

## 📝 Example Output

```bash
$ python sample.py
✅ Adaptive happy soundtrack generated and saved as adaptive_happy.mid
```

Play the generated file in any MIDI player or DAW (Digital Audio Workstation).

## 🎯 Use Cases

- **Game Development**: Dynamic background music
- **Content Creation**: YouTube videos, podcasts
- **Music Education**: Study AI composition
- **Prototyping**: Quick soundtrack mockups
- **Creative Exploration**: AI-assisted composition

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Support for chords/harmonies
- Rhythm pattern generation
- More sophisticated mood systems
- Real-time generation
- Plugin for DAWs

## 📄 License

MIT License - Feel free to use in personal and commercial projects!

## 🙏 Acknowledgments

- Built with PyTorch and music21
- LSTM architecture for sequence modeling
- MIDI format for universal compatibility

## 📧 Contact

**Author**: Akchhya Singh

---

⭐ **Star this repo if you found it helpful!**
