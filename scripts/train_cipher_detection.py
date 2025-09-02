#!/usr/bin/env python3
import pandas as pd, os, joblib, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter
import argparse

def extract_features(texts):
    # simple char frequency features for a-z and space
    features = []
    for s in texts:
        s = str(s).lower()
        cnt = Counter(c for c in s if 'a' <= c <= 'z' or c == ' ')
        total = sum(cnt.values()) or 1
        freqs = [cnt.get(chr(i+97),0)/total for i in range(26)]
        freqs.append(cnt.get(' ',0)/total)
        features.append(freqs + [len(s), len(set(s))])
    return np.array(features)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, default='models/cipher_detection_model.pkl')
    args = parser.parse_args()

    # load all datasets
    base = 'data'
    files = ['caesar_dataset.csv','vigenere_dataset.csv','substitution_dataset.csv']
    dfs = []
    for f in files:
        p = os.path.join(base,f)
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
    if not dfs:
        raise FileNotFoundError('No datasets found in data/')
    combined = pd.concat(dfs, ignore_index=True)
    X = extract_features(combined['encrypted_text'].astype(str).tolist())
    y = combined['cipher_type'].astype(str).values
    X_train, X_val, y_train, y_val = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    print('Accuracy:', accuracy_score(y_val,preds))
    print(classification_report(y_val,preds))
    os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
    joblib.dump(model, args.output)
    print('Saved cipher detection model to', args.output)

if __name__ == '__main__':
    main()
