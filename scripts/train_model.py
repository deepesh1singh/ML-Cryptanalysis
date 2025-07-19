#!/usr/bin/env python3
"""
Training Script for ML Cryptanalysis Models

This script trains machine learning models to classify and break classical ciphers
using frequency analysis and pattern recognition features.
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
import joblib
import logging
from typing import Dict, Any, Tuple
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_generator import DatasetGenerator
from feature_extractor import FeatureExtractor

# Set random seeds for reproducibility
np.random.seed(42)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """Train ML models for cryptanalysis"""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            ),
            'svm': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                random_state=42
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50),
                max_iter=500,
                random_state=42
            )
        }
        
        self.hyperparameters = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto'],
                'kernel': ['rbf', 'linear']
            },
            'neural_network': {
                'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                'alpha': [0.0001, 0.001, 0.01],
                'learning_rate': ['constant', 'adaptive']
            }
        }
        
        self.feature_extractor = FeatureExtractor()
        self.label_encoder = LabelEncoder()
        self.best_model = None
        self.best_params = None
    
    def load_data(self, cipher_type: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and prepare data for training
        
        Args:
            cipher_type: Type of cipher to load (None for combined)
        
        Returns:
            Feature matrix and labels
        """
        logger.info("Loading data...")
        
        if cipher_type:
            # Load specific cipher dataset
            data_path = f"data/{cipher_type}_dataset.csv"
            if not os.path.exists(data_path):
                logger.info(f"Generating {cipher_type} dataset...")
                generator = DatasetGenerator()
                
                if cipher_type == 'caesar':
                    df = generator.generate_caesar_dataset(1000)
                elif cipher_type == 'vigenere':
                    df = generator.generate_vigenere_dataset(1000)
                elif cipher_type == 'substitution':
                    df = generator.generate_substitution_dataset(1000)
                else:
                    raise ValueError(f"Unknown cipher type: {cipher_type}")
                
                generator.save_dataset(df, f"{cipher_type}_dataset.csv")
            else:
                df = pd.read_csv(data_path)
        else:
            # Load combined dataset
            data_path = "data/combined_dataset.csv"
            if not os.path.exists(data_path):
                logger.info("Generating combined dataset...")
                generator = DatasetGenerator()
                df = generator.generate_combined_dataset(500)
                generator.save_dataset(df, "combined_dataset.csv")
            else:
                df = pd.read_csv(data_path)
        
        # Extract features
        logger.info("Extracting features...")
        texts = df['encrypted_text'].tolist()
        X = self.feature_extractor.fit_transform(texts)
        
        # Prepare labels
        if cipher_type == 'caesar':
            # For Caesar cipher, predict the shift key
            y = df['key'].values
        else:
            # For other ciphers, predict cipher type
            y = self.label_encoder.fit_transform(df['cipher_type'].values)
        
        logger.info(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def train_model(self, model_name: str, X: np.ndarray, y: np.ndarray, 
                   tune_hyperparameters: bool = True) -> Dict[str, Any]:
        """
        Train a specific model
        
        Args:
            model_name: Name of the model to train
            X: Feature matrix
            y: Labels
            tune_hyperparameters: Whether to perform hyperparameter tuning
        
        Returns:
            Training results dictionary
        """
        logger.info(f"Training {model_name} model...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        if tune_hyperparameters and model_name in self.hyperparameters:
            # Perform hyperparameter tuning
            logger.info("Performing hyperparameter tuning...")
            grid_search = GridSearchCV(
                self.models[model_name],
                self.hyperparameters[model_name],
                cv=5,
                scoring='accuracy',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            logger.info(f"Best parameters: {best_params}")
        else:
            # Use default parameters
            best_model = self.models[model_name]
            best_model.fit(X_train, y_train)
            best_params = {}
        
        # Evaluate model
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(best_model, X, y, cv=5)
        
        results = {
            'model': best_model,
            'best_params': best_params,
            'accuracy': accuracy,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'y_test': y_test,
            'y_pred': y_pred
        }
        
        logger.info(f"{model_name} accuracy: {accuracy:.4f}")
        logger.info(f"Cross-validation score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return results
    
    def train_all_models(self, cipher_type: str = None, tune_hyperparameters: bool = True) -> Dict[str, Dict]:
        """
        Train all models and compare performance
        
        Args:
            cipher_type: Type of cipher to train on
            tune_hyperparameters: Whether to perform hyperparameter tuning
        
        Returns:
            Dictionary of training results for each model
        """
        # Load data
        X, y = self.load_data(cipher_type)
        
        results = {}
        
        for model_name in self.models.keys():
            try:
                result = self.train_model(model_name, X, y, tune_hyperparameters)
                results[model_name] = result
            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                continue
        
        return results
    
    def save_model(self, model, model_name: str, cipher_type: str = None):
        """
        Save trained model and related files
        
        Args:
            model: Trained model
            model_name: Name of the model
            cipher_type: Type of cipher
        """
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Save model
        model_filename = f"models/{cipher_type}_{model_name}_model.pkl"
        joblib.dump(model, model_filename)
        logger.info(f"Model saved to {model_filename}")
        
        # Save feature extractor
        scaler_filename = f"models/{cipher_type}_{model_name}_scaler.pkl"
        self.feature_extractor.save_scaler(scaler_filename)
        
        # Save label encoder if used
        if hasattr(self, 'label_encoder') and hasattr(self.label_encoder, 'classes_'):
            encoder_filename = f"models/{cipher_type}_{model_name}_encoder.pkl"
            joblib.dump(self.label_encoder, encoder_filename)
    
    def plot_results(self, results: Dict[str, Dict], cipher_type: str = None):
        """
        Plot training results
        
        Args:
            results: Training results dictionary
            cipher_type: Type of cipher
        """
        # Create results directory
        os.makedirs("results", exist_ok=True)
        
        # Plot accuracy comparison
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        cv_means = [results[name]['cv_mean'] for name in model_names]
        cv_stds = [results[name]['cv_std'] for name in model_names]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Accuracy comparison
        bars1 = ax1.bar(model_names, accuracies, alpha=0.8)
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom')
        
        # Cross-validation scores
        bars2 = ax2.bar(model_names, cv_means, yerr=cv_stds, alpha=0.8, capsize=5)
        ax2.set_title('Cross-Validation Scores')
        ax2.set_ylabel('CV Score')
        ax2.set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, mean, std in zip(bars2, cv_means, cv_stds):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{mean:.3f}±{std:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"results/{cipher_type}_model_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Plot confusion matrix for best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_result = results[best_model_name]
        
        cm = confusion_matrix(best_result['y_test'], best_result['y_pred'])
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {best_model_name.title()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f"results/{cipher_type}_{best_model_name}_confusion_matrix.png", 
                   dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description='Train ML models for cryptanalysis')
    parser.add_argument('--cipher', type=str, default=None,
                       choices=['caesar', 'vigenere', 'substitution'],
                       help='Type of cipher to train on (None for combined)')
    parser.add_argument('--model', type=str, default='all',
                       choices=['random_forest', 'svm', 'neural_network', 'all'],
                       help='Model to train')
    parser.add_argument('--tune', action='store_true',
                       help='Perform hyperparameter tuning')
    parser.add_argument('--plot', action='store_true',
                       help='Generate plots')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = ModelTrainer()
    
    # Train models
    if args.model == 'all':
        results = trainer.train_all_models(args.cipher, args.tune)
    else:
        X, y = trainer.load_data(args.cipher)
        results = {args.model: trainer.train_model(args.model, X, y, args.tune)}
    
    # Save best model
    best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
    best_result = results[best_model_name]
    
    trainer.save_model(best_result['model'], best_model_name, args.cipher or 'combined')
    
    # Generate plots if requested
    if args.plot:
        trainer.plot_results(results, args.cipher or 'combined')
    
    # Print summary
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    for model_name, result in results.items():
        print(f"{model_name.upper()}:")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  CV Score: {result['cv_mean']:.4f} (±{result['cv_std']:.4f})")
        if result['best_params']:
            print(f"  Best params: {result['best_params']}")
        print()
    
    print(f"Best model: {best_model_name}")
    print(f"Best accuracy: {best_result['accuracy']:.4f}")

if __name__ == "__main__":
    main() 