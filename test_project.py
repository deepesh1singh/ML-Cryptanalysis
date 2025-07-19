#!/usr/bin/env python3
"""
Test Script for ML Cryptanalysis Project

This script tests the basic functionality of the project components.
"""

import sys
import os
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_imports():
    """Test that all modules can be imported"""
    print("Testing imports...")
    
    try:
        from data_generator import DatasetGenerator, CaesarCipher, VigenereCipher, SubstitutionCipher
        print("✓ Data generator modules imported successfully")
    except ImportError as e:
        print(f"✗ Error importing data generator: {e}")
        return False
    
    try:
        from feature_extractor import FeatureExtractor
        print("✓ Feature extractor imported successfully")
    except ImportError as e:
        print(f"✗ Error importing feature extractor: {e}")
        return False
    
    return True

def test_cipher_implementations():
    """Test cipher implementations"""
    print("\nTesting cipher implementations...")
    
    # Test Caesar cipher
    from data_generator import CaesarCipher
    caesar = CaesarCipher()
    test_text = "hello world"
    key = 3
    encrypted = caesar.encrypt(test_text, key)
    decrypted = caesar.decrypt(encrypted, key)
    
    if decrypted == test_text.lower():
        print("✓ Caesar cipher working correctly")
    else:
        print("✗ Caesar cipher test failed")
        return False
    
    # Test Vigenère cipher
    from data_generator import VigenereCipher
    vigenere = VigenereCipher()
    test_text = "hello world"
    key = "key"
    encrypted = vigenere.encrypt(test_text, key)
    decrypted = vigenere.decrypt(encrypted, key)
    
    if decrypted == test_text.lower():
        print("✓ Vigenère cipher working correctly")
    else:
        print("✗ Vigenère cipher test failed")
        return False
    
    # Test substitution cipher
    from data_generator import SubstitutionCipher
    substitution = SubstitutionCipher()
    test_text = "hello world"
    key = substitution.generate_key()
    encrypted = substitution.encrypt(test_text, key)
    decrypted = substitution.decrypt(encrypted, key)
    
    if decrypted == test_text.lower():
        print("✓ Substitution cipher working correctly")
    else:
        print("✗ Substitution cipher test failed")
        return False
    
    return True

def test_data_generation():
    """Test data generation"""
    print("\nTesting data generation...")
    
    try:
        from data_generator import DatasetGenerator
        generator = DatasetGenerator()
        
        # Generate small datasets
        caesar_df = generator.generate_caesar_dataset(10)
        vigenere_df = generator.generate_vigenere_dataset(10)
        substitution_df = generator.generate_substitution_dataset(10)
        
        print(f"✓ Generated {len(caesar_df)} Caesar samples")
        print(f"✓ Generated {len(vigenere_df)} Vigenère samples")
        print(f"✓ Generated {len(substitution_df)} substitution samples")
        
        return True
    except Exception as e:
        print(f"✗ Data generation test failed: {e}")
        return False

def test_feature_extraction():
    """Test feature extraction"""
    print("\nTesting feature extraction...")
    
    try:
        from feature_extractor import FeatureExtractor
        extractor = FeatureExtractor()
        
        # Test texts
        test_texts = [
            "hello world this is a test",
            "the quick brown fox jumps over the lazy dog",
            "cryptography is fascinating"
        ]
        
        # Extract features
        feature_matrix = extractor.fit_transform(test_texts)
        feature_names = extractor.get_feature_names()
        
        print(f"✓ Extracted {feature_matrix.shape[1]} features from {feature_matrix.shape[0]} texts")
        print(f"✓ Feature names: {len(feature_names)} features")
        
        return True
    except Exception as e:
        print(f"✗ Feature extraction test failed: {e}")
        return False

def test_ml_components():
    """Test ML components"""
    print("\nTesting ML components...")
    
    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        # Generate test data
        from data_generator import DatasetGenerator
        generator = DatasetGenerator()
        df = generator.generate_caesar_dataset(100)
        
        # Extract features
        from feature_extractor import FeatureExtractor
        extractor = FeatureExtractor()
        X = extractor.fit_transform(df['encrypted_text'].tolist())
        y = df['key'].values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✓ ML pipeline working - accuracy: {accuracy:.4f}")
        
        return True
    except Exception as e:
        print(f"✗ ML components test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("="*50)
    print("ML CRYPTANALYSIS PROJECT TEST")
    print("="*50)
    
    tests = [
        test_imports,
        test_cipher_implementations,
        test_data_generation,
        test_feature_extraction,
        test_ml_components
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("="*50)
    print(f"TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Project is ready to use.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 