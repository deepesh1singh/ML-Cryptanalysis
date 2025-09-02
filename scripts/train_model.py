#!/usr/bin/env python3
import argparse, os, joblib
import pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
import xgboost as xgb
import optuna
import matplotlib.pyplot as plt

def load_dataset(path):
    df = pd.read_csv(path)
    if 'label' not in df.columns:
        # expect columns: features... and label
        # try to infer: last column as label
        df = df.copy()
        df['label'] = df.iloc[:, -1]
    X = df.drop('label', axis=1)
    y = df['label']
    return X, y

def build_model(name, **kwargs):
    if name == "random_forest":
        return RandomForestClassifier(random_state=42, **kwargs)
    if name == "svm":
        return SVC(probability=True, random_state=42, **kwargs)
    if name == "mlp":
        return MLPClassifier(max_iter=500, random_state=42, **kwargs)
    if name == "xgboost":
        return xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, **kwargs)
    raise ValueError("Unknown model")

def objective(trial, X_train, y_train, X_val, y_val, model_name):
    if model_name == "random_forest":
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 2, 20)
        model = build_model(model_name, n_estimators=n_estimators, max_depth=max_depth)
    elif model_name == "svm":
        C = trial.suggest_float("C", 0.1, 10.0, log=True)
        kernel = trial.suggest_categorical("kernel", ["linear","rbf"])
        model = build_model(model_name, C=C, kernel=kernel)
    elif model_name == "mlp":
        hidden_layer_sizes = trial.suggest_categorical("hidden_layer_sizes", [(100,),(200,),(100,100)])
        alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
        model = build_model(model_name, hidden_layer_sizes=hidden_layer_sizes, alpha=alpha)
    elif model_name == "xgboost":
        n_estimators = trial.suggest_int("n_estimators", 50, 300)
        max_depth = trial.suggest_int("max_depth", 2, 12)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        model = build_model(model_name, n_estimators=n_estimators, max_depth=max_depth,
                            learning_rate=learning_rate, subsample=subsample, colsample_bytree=colsample_bytree)
    else:
        raise ValueError("Model not supported for tuning")
    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    return accuracy_score(y_val, preds)

def save_report(report_text, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        f.write(report_text)

def main():
    parser = argparse.ArgumentParser(description="Train ML model for cryptanalysis")
    parser.add_argument("--cipher", type=str, required=True, help="Cipher type (caesar, vigenere, substitution)")
    parser.add_argument("--model", type=str, required=True, help="Model type (random_forest, svm, mlp, xgboost)")
    parser.add_argument("--tune", action="store_true", help="Use Optuna hyperparameter tuning")
    parser.add_argument("--plot", action="store_true", help="Plot feature importances (if supported)")
    args = parser.parse_args()

    data_path = f"data/{args.cipher}_dataset.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    X, y = load_dataset(data_path)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    if args.tune:
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: objective(trial, X_train, y_train, X_val, y_val, args.model), n_trials=30)
        best_params = study.best_params
        print("Best params:", best_params)
        model = build_model(args.model, **best_params)
    else:
        model = build_model(args.model)

    model.fit(X_train, y_train)
    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    report = classification_report(y_val, preds)
    cm = confusion_matrix(y_val, preds)
    print("Accuracy:", acc)
    print(report)
    print("Confusion matrix:\\n", cm)

    os.makedirs("models", exist_ok=True)
    model_path = f"models/{args.cipher}_{args.model}_model.pkl"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Save textual report
    save_report(f\"Accuracy: {acc}\\n\\n{report}\\n\\nConfusion matrix:\\n{cm}\", f\"results/{args.cipher}_{args.model}_evaluation.txt\")

    if args.plot and hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        idx = np.argsort(importances)[-20:]
        plt.figure(figsize=(8,6))
        plt.barh(range(len(idx)), importances[idx])
        plt.yticks(range(len(idx)), [X.columns[i] for i in idx])
        plt.title(f\"Top features ({args.model})\")
        plt.tight_layout()
        plt.savefig(f\"results/{args.cipher}_{args.model}_feature_importance.png\")
        print(\"Feature plot saved to results/\")


if __name__ == '__main__':
    main()
