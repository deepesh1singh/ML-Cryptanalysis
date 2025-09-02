#!/usr/bin/env python3
import argparse, os, joblib, sys
import pandas as pd
from collections import Counter
import numpy as np

# helper to extract features
def extract_features_single(s):
    s = str(s).lower()
    cnt = Counter(c for c in s if 'a' <= c <= 'z' or c == ' ')
    total = sum(cnt.values()) or 1
    freqs = [cnt.get(chr(i+97),0)/total for i in range(26)]
    freqs.append(cnt.get(' ',0)/total)
    return np.array(freqs + [len(s), len(set(s))]).reshape(1,-1)

def load_transformer(model_path):
    import torch, json
    state = torch.load(model_path, map_location='cpu')
    vocab = state.get('vocab')
    params = state.get('params',{})
    from scripts.train_transformer import SimpleTransformer
    model = SimpleTransformer(vocab_size=len(vocab)+1, d_model=params.get('d_model',128),
                              nhead=params.get('nhead',4), num_layers=params.get('num_layers',2))
    model.load_state_dict(state['model_state'])
    model.eval()
    return model, vocab

def transformer_decrypt(model, vocab, text, max_len=128):
    import torch
    stoi = vocab
    # encode
    arr = [stoi.get(c,0) for c in text[:max_len]]
    if len(arr)<max_len:
        arr += [0]*(max_len-len(arr))
    src = torch.tensor([arr], dtype=torch.long)
    out = model(src)  # batch, seq, vocab
    preds = out.argmax(dim=-1)[0].tolist()
    itos = {i:c for c,i in stoi.items()}
    decoded = ''.join(itos.get(p,'') for p in preds).strip()
    return decoded

def apply_classic_decrypt(cipher_type, ciphertext, key):
    # use src/data_generator cipher implementations
    sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
    from src.data_generator import CaesarCipher, VigenereCipher, SubstitutionCipher
    if cipher_type == 'caesar':
        c = CaesarCipher()
        return c.decrypt(ciphertext, int(key))
    if cipher_type == 'vigenere':
        c = VigenereCipher()
        return c.decrypt(ciphertext, str(key))
    if cipher_type == 'substitution':
        c = SubstitutionCipher()
        return c.decrypt(ciphertext, str(key))
    return None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='ciphertext string or path to file')
    parser.add_argument('--auto', action='store_true', help='auto-detect cipher type')
    parser.add_argument('--detector', type=str, default='models/cipher_detection_model.pkl')
    args = parser.parse_args()

    if os.path.exists(args.input):
        with open(args.input, 'r', encoding='utf-8') as f:
            ciphertext = f.read().strip()
    else:
        ciphertext = args.input

    if args.auto:
        if not os.path.exists(args.detector):
            print('Cipher detector not found. Train with: python scripts/train_cipher_detection.py')
            sys.exit(1)
        detector = joblib.load(args.detector)
        feats = extract_features_single(ciphertext)
        pred = detector.predict(feats)[0]
        print('Detected cipher:', pred)
        cipher = pred
    else:
        print('Auto not selected; please provide cipher type via --cipher in future.')
        sys.exit(1)

    # attempt to use transformer model first
    tm_path = f'models/{cipher}_transformer.pth'
    if os.path.exists(tm_path):
        try:
            model, vocab = load_transformer(tm_path)
            dec = transformer_decrypt(model, vocab, ciphertext)
            print('Transformer decryption (approx):', dec)
            return
        except Exception as e:
            print('Transformer decode failed:', e)

    # fallback: if classic model exists, predict key and apply decryption
    # assume model trained to predict 'key' values; try to load random forest model
    for candidate in [f'models/{cipher}_random_forest_model.pkl', f'models/{cipher}_xgboost_model.pkl', f'models/{cipher}_mlp_model.pkl']:
        if os.path.exists(candidate):
            m = joblib.load(candidate)
            # prepare a simple feature vector expected by training script (not strictly defined)
            try:
                # For classical models the training used structured features; here use character freqs
                feats = extract_features_single(ciphertext)
                key_pred = m.predict(feats)[0]
                print('Predicted key:', key_pred)
                dec = apply_classic_decrypt(cipher, ciphertext, key_pred)
                print('Decrypted text (using classic model predicted key):', dec)
                return
            except Exception as e:
                print('Failed using classic model:', e)
    print('No appropriate model found for decryption. Train models first.')

if __name__ == '__main__':
    main()
