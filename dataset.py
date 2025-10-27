import torch
from torch.utils.data import Dataset
import json

SEQ_LEN = 50

class NotesDataset(Dataset):
    def __init__(self, path="data/notes.json"):
        self.notes = json.load(open(path))
        self.unique = sorted(set(self.notes))
        self.note2idx = {n:i for i,n in enumerate(self.unique)}
        self.idx2note = {i:n for n,i in self.note2idx.items()}
        self.encoded = [self.note2idx[n] for n in self.notes]

    def __len__(self):
        return len(self.encoded) - SEQ_LEN

    def __getitem__(self, idx):
        seq = self.encoded[idx:idx+SEQ_LEN]
        target = self.encoded[idx+SEQ_LEN]
        return torch.tensor(seq), torch.tensor(target)
