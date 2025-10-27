import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset import NotesDataset
from model import MusicLSTM
from tqdm import tqdm

ds = NotesDataset()
dl = DataLoader(ds, batch_size=64, shuffle=True)
model = MusicLSTM(vocab_size=len(ds.unique))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

EPOCHS = 20
for e in range(EPOCHS):
    total = 0
    for x, y in tqdm(dl):
        logits, _ = model(x)
        loss = F.cross_entropy(logits, y)
        opt.zero_grad(); loss.backward(); opt.step()
        total += loss.item()
    print(f"Epoch {e+1}/{EPOCHS}, Loss={total/len(dl):.4f}")

torch.save({
    "model": model.state_dict(),
    "vocab": ds.unique
}, "music_model.pt")
print("✅ Model saved to music_model.pt")
