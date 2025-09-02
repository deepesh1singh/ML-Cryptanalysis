#!/usr/bin/env python3
import argparse, os, joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def load_dataset(path):
    df = pd.read_csv(path)
    if 'label' not in df.columns:
        df['label'] = df.iloc[:, -1]
    X = df.drop('label', axis=1)
    y = df['label']
    return X, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--cipher_type', type=str, required=True)
    parser.add_argument('--test_samples', type=int, default=500)
    args = parser.parse_args()

    model = joblib.load(args.model_path)
    data_path = f\"data/{args.cipher_type}_dataset.csv\"
    X, y = load_dataset(data_path)
    # sample
    if args.test_samples < len(X):
        X = X.sample(n=args.test_samples, random_state=42)
        y = y.loc[X.index]

    preds = model.predict(X)
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X)
    else:
        probs = None

    acc = accuracy_score(y, preds)
    print('Accuracy:', acc)
    print(classification_report(y, preds))
    cm = confusion_matrix(y, preds)
    os.makedirs('results', exist_ok=True)
    # plot confusion matrix
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(f'results/{args.cipher_type}_confusion_matrix.png')
    print('Saved confusion matrix to results/')
    # Save report
    with open(f'results/{args.cipher_type}_evaluation_report.txt','w') as f:
        f.write(f'Accuracy: {acc}\\n\\n')
        f.write(classification_report(y, preds))
    print('Saved textual report.')

if __name__ == '__main__':
    main()
