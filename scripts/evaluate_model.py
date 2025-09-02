#!/usr/bin/env python3
"""
Evaluation Script for ML Cryptanalysis Models

This script evaluates trained models on test data and provides detailed
performance metrics and visualizations.
"""

import argparse
import sys
import os
import numpy as np
import pandas as pd
import joblib
import logging
from typing import Dict, Any, Tuple, List
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_generator import DatasetGenerator
from feature_extractor import FeatureExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Evaluate ML models for cryptanalysis"""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        self.results = {}
    
    def load_model(self, model_path: str) -> Tuple[Any, Any, Any]:
        """
        Load trained model and related files
        
        Args:
            model_path: Path to the model file
        
        Returns:
            Tuple of (model, scaler, encoder)
        """
        # Load model
        model = joblib.load(model_path)
        logger.info(f"Model loaded from {model_path}")
        
        # Load scaler
        scaler_path = model_path.replace('_model.pkl', '_scaler.pkl')
        scaler = None
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded from {scaler_path}")
        
        # Load encoder
        encoder_path = model_path.replace('_model.pkl', '_encoder.pkl')
        encoder = None
        if os.path.exists(encoder_path):
            encoder = joblib.load(encoder_path)
            logger.info(f"Encoder loaded from {encoder_path}")
        
        return model, scaler, encoder
    
    def generate_test_data(self, cipher_type: str, num_samples: int = 200) -> pd.DataFrame:
        """
        Generate test data for evaluation
        
        Args:
            cipher_type: Type of cipher
            num_samples: Number of test samples
        
        Returns:
            DataFrame with test data
        """
        generator = DatasetGenerator()
        
        if cipher_type == 'caesar':
            df = generator.generate_caesar_dataset(num_samples)
        elif cipher_type == 'vigenere':
            df = generator.generate_vigenere_dataset(num_samples)
        elif cipher_type == 'substitution':
            df = generator.generate_substitution_dataset(num_samples)
        else:
            raise ValueError(f"Unknown cipher type: {cipher_type}")
        
        return df
    
    def evaluate_model(self, model_path: str, cipher_type: str = None, 
                      num_test_samples: int = 200) -> Dict[str, Any]:
        """
        Evaluate a trained model
        
        Args:
            model_path: Path to the model file
            cipher_type: Type of cipher (inferred from path if None)
            num_test_samples: Number of test samples to generate
        
        Returns:
            Evaluation results dictionary
        """
        # Load model and related files
        model, scaler, encoder = self.load_model(model_path)
        
        # Infer cipher type from model path if not provided
        if cipher_type is None:
            model_name = os.path.basename(model_path)
            if 'caesar' in model_name:
                cipher_type = 'caesar'
            elif 'vigenere' in model_name:
                cipher_type = 'vigenere'
            elif 'substitution' in model_name:
                cipher_type = 'substitution'
            else:
                cipher_type = 'combined'
        
        # Generate test data
        logger.info(f"Generating {num_test_samples} test samples for {cipher_type} cipher...")
        test_df = self.generate_test_data(cipher_type, num_test_samples)
        
        # Extract features
        test_texts = test_df['encrypted_text'].tolist()
        
        if scaler is not None:
            # Use loaded scaler
            self.feature_extractor.scaler = scaler
            self.feature_extractor.is_fitted = True
            X_test = self.feature_extractor.transform(test_texts)
        else:
            # Fit new scaler (fallback)
            X_test = self.feature_extractor.fit_transform(test_texts)
        
        # Prepare labels
        if cipher_type == 'caesar':
            y_test = test_df['key'].values
        else:
            if encoder is not None:
                y_test = encoder.transform(test_df['cipher_type'].values)
            else:
                # Create simple encoding
                unique_types = test_df['cipher_type'].unique()
                type_to_idx = {t: i for i, t in enumerate(unique_types)}
                y_test = [type_to_idx[t] for t in test_df['cipher_type'].values]
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Classification report
        if cipher_type == 'caesar':
            target_names = [f'key_{i}' for i in range(1, 26)]
        else:
            target_names = test_df['cipher_type'].unique() if encoder is None else encoder.classes_
        
        report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Precision, recall, F1-score
        precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
        
        # ROC AUC (if applicable)
        roc_auc = None
        if y_pred_proba is not None and len(np.unique(y_test)) == 2:
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        results = {
            'model_path': model_path,
            'cipher_type': cipher_type,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'classification_report': report,
            'confusion_matrix': cm,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba,
            'test_data': test_df,
            'feature_matrix': X_test
        }
        
        logger.info(f"Evaluation completed:")
        logger.info(f"  Accuracy: {accuracy:.4f}")
        logger.info(f"  Precision: {precision:.4f}")
        logger.info(f"  Recall: {recall:.4f}")
        logger.info(f"  F1-Score: {f1:.4f}")
        if roc_auc:
            logger.info(f"  ROC AUC: {roc_auc:.4f}")
        
        return results
    
    def plot_evaluation_results(self, results: Dict[str, Any], save_plots: bool = True):
        """
        Plot evaluation results
        
        Args:
            results: Evaluation results dictionary
            save_plots: Whether to save plots to files
        """
        cipher_type = results['cipher_type']
        
        # Create results directory
        if save_plots:
            os.makedirs("results", exist_ok=True)
        
        # 1. Confusion Matrix
        plt.figure(figsize=(10, 8))
        cm = results['confusion_matrix']
        
        if cipher_type == 'caesar':
            # For Caesar cipher, show key predictions
            labels = [f'Key {i}' for i in range(1, 26)]
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=labels, yticklabels=labels)
            plt.title(f'Confusion Matrix - {cipher_type.title()} Cipher Key Prediction')
        else:
            # For other ciphers, show cipher type classification
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {cipher_type.title()} Cipher Classification')
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        if save_plots:
            plt.savefig(f"results/{cipher_type}_confusion_matrix.png", 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Feature Importance (for Random Forest)
        if hasattr(results.get('model'), 'feature_importances_'):
            model = joblib.load(results['model_path'])
            if hasattr(model, 'feature_importances_'):
                feature_names = self.feature_extractor.get_feature_names()
                importances = model.feature_importances_
                
                # Get top 20 features
                indices = np.argsort(importances)[-20:]
                
                plt.figure(figsize=(12, 8))
                plt.barh(range(len(indices)), importances[indices])
                plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
                plt.xlabel('Feature Importance')
                plt.title(f'Top 20 Feature Importances - {cipher_type.title()} Cipher')
                plt.tight_layout()
                
                if save_plots:
                    plt.savefig(f"results/{cipher_type}_feature_importance.png", 
                               dpi=300, bbox_inches='tight')
                plt.show()
        
        # 3. Prediction Distribution
        plt.figure(figsize=(10, 6))
        
        if cipher_type == 'caesar':
            # Show key prediction distribution
            key_counts = Counter(results['y_pred'])
            keys = sorted(key_counts.keys())
            counts = [key_counts[k] for k in keys]
            
            plt.bar(keys, counts, alpha=0.7)
            plt.xlabel('Predicted Key')
            plt.ylabel('Number of Predictions')
            plt.title(f'Key Prediction Distribution - {cipher_type.title()} Cipher')
        else:
            # Show cipher type prediction distribution
            pred_counts = Counter(results['y_pred'])
            plt.bar(pred_counts.keys(), pred_counts.values(), alpha=0.7)
            plt.xlabel('Predicted Cipher Type')
            plt.ylabel('Number of Predictions')
            plt.title(f'Cipher Type Prediction Distribution - {cipher_type.title()} Cipher')
        
        if save_plots:
            plt.savefig(f"results/{cipher_type}_prediction_distribution.png", 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
        # 4. Accuracy by Text Length
        test_df = results['test_data']
        text_lengths = test_df['encrypted_text'].str.len()
        
        # Group by text length ranges
        length_ranges = pd.cut(text_lengths, bins=5)
        accuracy_by_length = []
        
        for length_range in length_ranges.unique():
            if pd.isna(length_range):
                continue
            mask = length_ranges == length_range
            if mask.sum() > 0:
                subset_accuracy = accuracy_score(
                    np.array(results['y_test'])[mask],
                    np.array(results['y_pred'])[mask]
                )
                accuracy_by_length.append((str(length_range), subset_accuracy))
        
        if accuracy_by_length:
            ranges, accuracies = zip(*accuracy_by_length)
            
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(ranges)), accuracies, alpha=0.7)
            plt.xlabel('Text Length Range')
            plt.ylabel('Accuracy')
            plt.title(f'Accuracy by Text Length - {cipher_type.title()} Cipher')
            plt.xticks(range(len(ranges)), ranges, rotation=45)
            
            if save_plots:
                plt.savefig(f"results/{cipher_type}_accuracy_by_length.png", 
                           dpi=300, bbox_inches='tight')
            plt.show()
    
    def generate_evaluation_report(self, results: Dict[str, Any], output_file: str = None):
        """
        Generate a detailed evaluation report
        
        Args:
            results: Evaluation results dictionary
            output_file: Path to save the report (optional)
        """
        report_lines = []
        report_lines.append("="*60)
        report_lines.append("ML CRYPTANALYSIS MODEL EVALUATION REPORT")
        report_lines.append("="*60)
        report_lines.append("")
        
        # Model information
        report_lines.append("MODEL INFORMATION:")
        report_lines.append(f"  Model Path: {results['model_path']}")
        report_lines.append(f"  Cipher Type: {results['cipher_type']}")
        report_lines.append(f"  Test Samples: {len(results['y_test'])}")
        report_lines.append("")
        
        # Performance metrics
        report_lines.append("PERFORMANCE METRICS:")
        report_lines.append(f"  Accuracy: {results['accuracy']:.4f}")
        report_lines.append(f"  Precision: {results['precision']:.4f}")
        report_lines.append(f"  Recall: {results['recall']:.4f}")
        report_lines.append(f"  F1-Score: {results['f1_score']:.4f}")
        if results['roc_auc']:
            report_lines.append(f"  ROC AUC: {results['roc_auc']:.4f}")
        report_lines.append("")
        
        # Detailed classification report
        report_lines.append("DETAILED CLASSIFICATION REPORT:")
        report_lines.append("")
        
        # Convert classification report to string
        report_dict = results['classification_report']
        for class_name, metrics in report_dict.items():
            if isinstance(metrics, dict):
                report_lines.append(f"  {class_name}:")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        report_lines.append(f"    {metric}: {value:.4f}")
                    else:
                        report_lines.append(f"    {metric}: {value}")
                report_lines.append("")
        
        # Confusion matrix summary
        cm = results['confusion_matrix']
        report_lines.append("CONFUSION MATRIX SUMMARY:")
        report_lines.append(f"  Total Predictions: {cm.sum()}")
        report_lines.append(f"  Correct Predictions: {cm.trace()}")
        report_lines.append(f"  Incorrect Predictions: {cm.sum() - cm.trace()}")
        report_lines.append("")
        
        # Save report
        report_text = "\n".join(report_lines)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            logger.info(f"Evaluation report saved to {output_file}")
        
        # Print report
        print(report_text)

def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description='Evaluate ML models for cryptanalysis')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model file')
    parser.add_argument('--cipher_type', type=str, default=None,
                       choices=['caesar', 'vigenere', 'substitution'],
                       help='Type of cipher (inferred from path if not provided)')
    parser.add_argument('--test_samples', type=int, default=200,
                       help='Number of test samples to generate')
    parser.add_argument('--plot', action='store_true',
                       help='Generate evaluation plots')
    parser.add_argument('--report', type=str, default=None,
                       help='Path to save evaluation report')
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Evaluate model
    try:
        results = evaluator.evaluate_model(
            args.model_path, 
            args.cipher_type, 
            args.test_samples
        )
        
        # Generate plots if requested
        if args.plot:
            evaluator.plot_evaluation_results(results)
        
        # Generate report
        report_path = args.report or f"results/{results['cipher_type']}_evaluation_report.txt"
        evaluator.generate_evaluation_report(results, report_path)
        
        # Print summary
        print("\n" + "="*50)
        print("EVALUATION SUMMARY")
        print("="*50)
        print(f"Model: {os.path.basename(args.model_path)}")
        print(f"Cipher Type: {results['cipher_type']}")
        print(f"Accuracy: {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1-Score: {results['f1_score']:.4f}")
        if results['roc_auc']:
            print(f"ROC AUC: {results['roc_auc']:.4f}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 