"""
Feature Extractor for Cryptanalysis

This module extracts features from encrypted text for machine learning
analysis, including character frequency analysis and n-gram features.
"""

import numpy as np
import pandas as pd
import string
from collections import Counter
from typing import List, Dict, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extract features from encrypted text for ML analysis"""
    
    def __init__(self):
        self.alphabet = string.ascii_lowercase
        self.english_freq = {
            'a': 8.2, 'b': 1.5, 'c': 2.8, 'd': 4.3, 'e': 13.0,
            'f': 2.2, 'g': 2.0, 'h': 6.1, 'i': 7.0, 'j': 0.15,
            'k': 0.77, 'l': 4.0, 'm': 2.4, 'n': 6.7, 'o': 7.5,
            'p': 1.9, 'q': 0.095, 'r': 6.0, 's': 6.3, 't': 9.1,
            'u': 2.8, 'v': 0.98, 'w': 2.4, 'x': 0.15, 'y': 2.0, 'z': 0.074
        }
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def extract_character_frequencies(self, text: str) -> Dict[str, float]:
        """
        Extract character frequency features
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of character frequencies
        """
        # Count characters
        char_count = Counter(text.lower())
        total_chars = sum(char_count.values())
        
        # Calculate frequencies
        frequencies = {}
        for char in self.alphabet:
            count = char_count.get(char, 0)
            frequencies[f'freq_{char}'] = count / total_chars if total_chars > 0 else 0
        
        return frequencies
    
    def extract_bigram_frequencies(self, text: str) -> Dict[str, float]:
        """
        Extract bigram frequency features
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of bigram frequencies
        """
        # Generate bigrams
        bigrams = []
        for i in range(len(text) - 1):
            if text[i].isalpha() and text[i+1].isalpha():
                bigrams.append(text[i:i+2].lower())
        
        # Count bigrams
        bigram_count = Counter(bigrams)
        total_bigrams = sum(bigram_count.values())
        
        # Get top bigrams
        top_bigrams = ['th', 'he', 'an', 'in', 'er', 're', 'on', 'at', 'nd', 'st']
        frequencies = {}
        
        for bigram in top_bigrams:
            count = bigram_count.get(bigram, 0)
            frequencies[f'bigram_{bigram}'] = count / total_bigrams if total_bigrams > 0 else 0
        
        return frequencies
    
    def extract_trigram_frequencies(self, text: str) -> Dict[str, float]:
        """
        Extract trigram frequency features
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of trigram frequencies
        """
        # Generate trigrams
        trigrams = []
        for i in range(len(text) - 2):
            if text[i].isalpha() and text[i+1].isalpha() and text[i+2].isalpha():
                trigrams.append(text[i:i+3].lower())
        
        # Count trigrams
        trigram_count = Counter(trigrams)
        total_trigrams = sum(trigram_count.values())
        
        # Get top trigrams
        top_trigrams = ['the', 'and', 'ing', 'ion', 'tio', 'for', 'nde', 'has', 'nce', 'edt']
        frequencies = {}
        
        for trigram in top_trigrams:
            count = trigram_count.get(trigram, 0)
            frequencies[f'trigram_{trigram}'] = count / total_trigrams if total_trigrams > 0 else 0
        
        return frequencies
    
    def extract_statistical_features(self, text: str) -> Dict[str, float]:
        """
        Extract statistical features
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of statistical features
        """
        features = {}
        
        # Text length
        features['text_length'] = len(text)
        
        # Character count
        features['char_count'] = sum(1 for c in text if c.isalpha())
        
        # Space count
        features['space_count'] = text.count(' ')
        
        # Vowel count
        vowels = 'aeiou'
        features['vowel_count'] = sum(1 for c in text.lower() if c in vowels)
        
        # Consonant count
        consonants = 'bcdfghjklmnpqrstvwxyz'
        features['consonant_count'] = sum(1 for c in text.lower() if c in consonants)
        
        # Vowel to consonant ratio
        if features['consonant_count'] > 0:
            features['vowel_consonant_ratio'] = features['vowel_count'] / features['consonant_count']
        else:
            features['vowel_consonant_ratio'] = 0
        
        # Average word length
        words = text.split()
        if words:
            features['avg_word_length'] = sum(len(word) for word in words) / len(words)
        else:
            features['avg_word_length'] = 0
        
        # Unique character ratio
        unique_chars = len(set(text.lower()))
        features['unique_char_ratio'] = unique_chars / len(text) if len(text) > 0 else 0
        
        return features
    
    def extract_entropy_features(self, text: str) -> Dict[str, float]:
        """
        Extract entropy-based features
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of entropy features
        """
        features = {}
        
        # Character entropy
        char_count = Counter(text.lower())
        total_chars = sum(char_count.values())
        
        if total_chars > 0:
            entropy = 0
            for count in char_count.values():
                p = count / total_chars
                if p > 0:
                    entropy -= p * np.log2(p)
            features['char_entropy'] = entropy
        else:
            features['char_entropy'] = 0
        
        # Bigram entropy
        bigrams = []
        for i in range(len(text) - 1):
            if text[i].isalpha() and text[i+1].isalpha():
                bigrams.append(text[i:i+2].lower())
        
        bigram_count = Counter(bigrams)
        total_bigrams = sum(bigram_count.values())
        
        if total_bigrams > 0:
            bigram_entropy = 0
            for count in bigram_count.values():
                p = count / total_bigrams
                if p > 0:
                    bigram_entropy -= p * np.log2(p)
            features['bigram_entropy'] = bigram_entropy
        else:
            features['bigram_entropy'] = 0
        
        return features
    
    def extract_frequency_deviation(self, text: str) -> Dict[str, float]:
        """
        Extract frequency deviation from English language
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of frequency deviation features
        """
        features = {}
        
        # Get character frequencies
        char_freq = self.extract_character_frequencies(text)
        
        # Calculate deviation from English frequencies
        total_deviation = 0
        for char in self.alphabet:
            freq_key = f'freq_{char}'
            actual_freq = char_freq[freq_key]
            expected_freq = self.english_freq[char] / 100  # Convert to decimal
            
            deviation = abs(actual_freq - expected_freq)
            features[f'dev_{char}'] = deviation
            total_deviation += deviation
        
        features['total_frequency_deviation'] = total_deviation
        features['avg_frequency_deviation'] = total_deviation / 26
        
        return features
    
    def extract_all_features(self, text: str) -> Dict[str, float]:
        """
        Extract all features from text
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of all features
        """
        features = {}
        
        # Extract all feature types
        features.update(self.extract_character_frequencies(text))
        features.update(self.extract_bigram_frequencies(text))
        features.update(self.extract_trigram_frequencies(text))
        features.update(self.extract_statistical_features(text))
        features.update(self.extract_entropy_features(text))
        features.update(self.extract_frequency_deviation(text))
        
        return features
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        Fit the scaler and transform texts to feature matrix
        
        Args:
            texts: List of input texts
            
        Returns:
            Feature matrix
        """
        # Extract features for all texts
        feature_dicts = [self.extract_all_features(text) for text in texts]
        
        # Convert to DataFrame
        feature_df = pd.DataFrame(feature_dicts)
        
        # Fit scaler and transform
        feature_matrix = self.scaler.fit_transform(feature_df)
        self.is_fitted = True
        
        return feature_matrix
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to feature matrix (after fitting)
        
        Args:
            texts: List of input texts
            
        Returns:
            Feature matrix
        """
        if not self.is_fitted:
            raise ValueError("Must call fit_transform first")
        
        # Extract features for all texts
        feature_dicts = [self.extract_all_features(text) for text in texts]
        
        # Convert to DataFrame
        feature_df = pd.DataFrame(feature_dicts)
        
        # Transform using fitted scaler
        feature_matrix = self.scaler.transform(feature_df)
        
        return feature_matrix
    
    def get_feature_names(self) -> List[str]:
        """
        Get list of feature names
        
        Returns:
            List of feature names
        """
        # Create a sample text to get feature names
        sample_text = "the quick brown fox jumps over the lazy dog"
        features = self.extract_all_features(sample_text)
        return list(features.keys())
    
    def save_scaler(self, filepath: str):
        """
        Save the fitted scaler
        
        Args:
            filepath: Path to save scaler
        """
        if self.is_fitted:
            import joblib
            joblib.dump(self.scaler, filepath)
            logger.info(f"Scaler saved to {filepath}")
        else:
            raise ValueError("Scaler not fitted yet")
    
    def load_scaler(self, filepath: str):
        """
        Load a fitted scaler
        
        Args:
            filepath: Path to load scaler from
        """
        import joblib
        self.scaler = joblib.load(filepath)
        self.is_fitted = True
        logger.info(f"Scaler loaded from {filepath}")

def main():
    """Test feature extraction"""
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
    
    print(f"Feature matrix shape: {feature_matrix.shape}")
    print(f"Number of features: {len(feature_names)}")
    print(f"First few features: {feature_names[:10]}")
    
    # Save scaler
    extractor.save_scaler("models/feature_scaler.pkl")

if __name__ == "__main__":
    main() 