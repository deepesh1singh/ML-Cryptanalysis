#!/usr/bin/env python3
"""Character-level Transformer for ciphertext -> plaintext mapping using dataset columns:
   - original_text (plaintext)
   - encrypted_text (ciphertext)
This script trains a simple Transformer encoder to predict plaintext from ciphertext.
"""
import argparse, os, json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class CharDataset(Dataset):
    def __init__(self, df, src_col='encrypted_text', tgt_col='original_text', max_len=128, charset=None):
        self.src = df[src_col].astype(str).tolist()
        self.tgt = df[tgt_col].astype(str).tolist()
        self.max_len = max_len
        if charset is None:
            chars = set(''.join(self.src + self.tgt))
            # ensure space included
            chars.add(' ')
            self.chars = sorted(list(chars))
        else:
            self.chars = charset
        self.stoi = {c:i+1 for i,c in enumerate(self.chars)}  # 0 reserved for padding
        self.itos = {i:c for c,i in self.stoi.items()}

    def encode(self, s):
        arr = [self.stoi.get(c,0) for c in s[:self.max_len]]
        if len(arr) < self.max_len:
            arr += [0]*(self.max_len - len(arr))
        return torch.tensor(arr, dtype=torch.long)

    def __len__(self):
        return len(self.src)

    def __getitem__(self, idx):
        return self.encode(self.src[idx]), self.encode(self.tgt[idx])

class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        x = self.emb(src).permute(1,0,2)  # seq_len, batch, dim
        h = self.encoder(x)
        h = h.permute(1,0,2)  # batch, seq, dim
        return self.fc(h)

def collate_fn(batch):
    srcs = torch.stack([b[0] for b in batch])
    tgts = torch.stack([b[1] for b in batch])
    return srcs, tgts

def train(args):
    data = pd.read_csv(args.data_path)
    if 'original_text' not in data.columns or 'encrypted_text' not in data.columns:
        raise ValueError(\"Dataset must contain 'original_text' and 'encrypted_text' columns for seq2seq training.\")
    # optionally filter by cipher type
    if args.cipher and 'cipher_type' in data.columns:
        data = data[data['cipher_type']==args.cipher].reset_index(drop=True)
    ds = CharDataset(data, max_len=args.max_len)
    vocab_size = len(ds.stoi)+1
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    model = SimpleTransformer(vocab_size, d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers)
    device = torch.device('cuda' if torch.cuda.is_available() and args.device=='cuda' else 'cpu')
    model.to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for src, tgt in dl:
            src = src.to(device)
            tgt = tgt.to(device)
            optim.zero_grad()
            out = model(src) # batch, seq, vocab
            loss = loss_fn(out.view(-1, out.size(-1)), tgt.view(-1))
            loss.backward()
            optim.step()
            total_loss += loss.item()
        print(f\"Epoch {epoch+1}/{args.epochs}, loss={total_loss/len(dl):.4f}\")
    os.makedirs('models', exist_ok=True)
    save_path = f\"models/{args.cipher}_transformer.pth\"
    torch.save({'model_state': model.state_dict(), 'vocab': ds.stoi, 'params': vars(args)}, save_path)
    print(\"Saved transformer model to\", save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data/substitution_dataset.csv')
    parser.add_argument('--cipher', type=str, default='substitution')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--nhead', type=int, default=4)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()
    train(args)
