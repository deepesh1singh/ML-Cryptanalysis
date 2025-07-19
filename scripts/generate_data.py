#!/usr/bin/env python3
"""
Data Generation Script for ML Cryptanalysis

This script generates encrypted text datasets for training and testing
machine learning models for cryptanalysis.
"""

import argparse
import sys
import os
import logging
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_generator import DatasetGenerator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Generate datasets for ML cryptanalysis"""
    parser = argparse.ArgumentParser(description='Generate datasets for ML cryptanalysis')
    parser.add_argument('--cipher', type=str, default='all',
                       choices=['caesar', 'vigenere', 'substitution', 'all'],
                       help='Type of cipher to generate data for')
    parser.add_argument('--samples', type=int, default=1000,
                       help='Number of samples to generate per cipher type')
    parser.add_argument('--output-dir', type=str, default='data',
                       help='Output directory for datasets')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize generator
    generator = DatasetGenerator()
    
    # Generate datasets based on specified cipher type
    if args.cipher == 'all':
        logger.info("Generating all datasets...")
        
        # Generate individual datasets
        logger.info(f"Generating Caesar cipher dataset ({args.samples} samples)...")
        caesar_df = generator.generate_caesar_dataset(args.samples)
        generator.save_dataset(caesar_df, "caesar_dataset.csv")
        
        logger.info(f"Generating Vigenère cipher dataset ({args.samples} samples)...")
        vigenere_df = generator.generate_vigenere_dataset(args.samples)
        generator.save_dataset(vigenere_df, "vigenere_dataset.csv")
        
        logger.info(f"Generating substitution cipher dataset ({args.samples} samples)...")
        substitution_df = generator.generate_substitution_dataset(args.samples)
        generator.save_dataset(substitution_df, "substitution_dataset.csv")
        
        # Generate combined dataset
        logger.info("Generating combined dataset...")
        combined_df = generator.generate_combined_dataset(args.samples // 2)
        generator.save_dataset(combined_df, "combined_dataset.csv")
        
    elif args.cipher == 'caesar':
        logger.info(f"Generating Caesar cipher dataset ({args.samples} samples)...")
        df = generator.generate_caesar_dataset(args.samples)
        generator.save_dataset(df, "caesar_dataset.csv")
        
    elif args.cipher == 'vigenere':
        logger.info(f"Generating Vigenère cipher dataset ({args.samples} samples)...")
        df = generator.generate_vigenere_dataset(args.samples)
        generator.save_dataset(df, "vigenere_dataset.csv")
        
    elif args.cipher == 'substitution':
        logger.info(f"Generating substitution cipher dataset ({args.samples} samples)...")
        df = generator.generate_substitution_dataset(args.samples)
        generator.save_dataset(df, "substitution_dataset.csv")
    
    logger.info("Data generation completed successfully!")
    
    # Print dataset statistics
    print("\n" + "="*50)
    print("DATASET STATISTICS")
    print("="*50)
    
    if args.cipher == 'all':
        datasets = [
            ("Caesar", "caesar_dataset.csv"),
            ("Vigenère", "vigenere_dataset.csv"),
            ("Substitution", "substitution_dataset.csv"),
            ("Combined", "combined_dataset.csv")
        ]
    else:
        datasets = [(args.cipher.title(), f"{args.cipher}_dataset.csv")]
    
    for name, filename in datasets:
        filepath = os.path.join(args.output_dir, filename)
        if os.path.exists(filepath):
            import pandas as pd
            df = pd.read_csv(filepath)
            print(f"{name} Dataset:")
            print(f"  Samples: {len(df)}")
            print(f"  Columns: {list(df.columns)}")
            if 'key' in df.columns:
                print(f"  Key range: {df['key'].min()} to {df['key'].max()}")
            if 'cipher_type' in df.columns:
                print(f"  Cipher types: {df['cipher_type'].unique()}")
            print()

if __name__ == "__main__":
    main() 