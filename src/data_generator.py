"""
Data Generator for Classical Ciphers

This module generates encrypted text samples using classical ciphers
for machine learning-based cryptanalysis.
"""

import random
import string
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import logging

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CipherGenerator:
    """Base class for cipher implementations"""
    
    def __init__(self):
        self.alphabet = string.ascii_lowercase
        self.alphabet_size = len(self.alphabet)
    
    def encrypt(self, text: str, key: str) -> str:
        """Encrypt text using the cipher"""
        raise NotImplementedError
    
    def decrypt(self, text: str, key: str) -> str:
        """Decrypt text using the cipher"""
        raise NotImplementedError

class CaesarCipher(CipherGenerator):
    """Implementation of Caesar Cipher (shift cipher)"""
    
    def encrypt(self, text: str, key: int) -> str:
        """
        Encrypt text using Caesar cipher
        
        Args:
            text: Plain text to encrypt
            key: Shift value (0-25)
        
        Returns:
            Encrypted text
        """
        encrypted = ""
        for char in text.lower():
            if char.isalpha():
                # Apply shift and wrap around alphabet
                shifted = (ord(char) - ord('a') + key) % 26
                encrypted += chr(shifted + ord('a'))
            else:
                encrypted += char
        return encrypted
    
    def decrypt(self, text: str, key: int) -> str:
        """
        Decrypt text using Caesar cipher
        
        Args:
            text: Encrypted text
            key: Shift value (0-25)
        
        Returns:
            Decrypted text
        """
        return self.encrypt(text, -key)

class VigenereCipher(CipherGenerator):
    """Implementation of Vigenère Cipher (polyalphabetic substitution)"""
    
    def encrypt(self, text: str, key: str) -> str:
        """
        Encrypt text using Vigenère cipher
        
        Args:
            text: Plain text to encrypt
            key: Keyword for encryption
        
        Returns:
            Encrypted text
        """
        encrypted = ""
        key = key.lower()
        key_length = len(key)
        
        for i, char in enumerate(text.lower()):
            if char.isalpha():
                # Get shift from key
                key_char = key[i % key_length]
                shift = ord(key_char) - ord('a')
                
                # Apply shift
                shifted = (ord(char) - ord('a') + shift) % 26
                encrypted += chr(shifted + ord('a'))
            else:
                encrypted += char
        return encrypted
    
    def decrypt(self, text: str, key: str) -> str:
        """
        Decrypt text using Vigenère cipher
        
        Args:
            text: Encrypted text
            key: Keyword for decryption
        
        Returns:
            Decrypted text
        """
        decrypted = ""
        key = key.lower()
        key_length = len(key)
        
        for i, char in enumerate(text.lower()):
            if char.isalpha():
                # Get shift from key
                key_char = key[i % key_length]
                shift = ord(key_char) - ord('a')
                
                # Apply reverse shift
                shifted = (ord(char) - ord('a') - shift) % 26
                decrypted += chr(shifted + ord('a'))
            else:
                decrypted += char
        return decrypted

class SubstitutionCipher(CipherGenerator):
    """Implementation of Substitution Cipher (monoalphabetic substitution)"""
    
    def __init__(self):
        super().__init__()
        self.substitution_map = {}
        self.reverse_map = {}
    
    def generate_key(self) -> str:
        """Generate a random substitution key"""
        # Create a shuffled alphabet
        shuffled = list(self.alphabet)
        random.shuffle(shuffled)
        return ''.join(shuffled)
    
    def set_key(self, key: str):
        """Set the substitution key"""
        if len(key) != 26:
            raise ValueError("Key must be 26 characters long")
        
        self.substitution_map = dict(zip(self.alphabet, key))
        self.reverse_map = dict(zip(key, self.alphabet))
    
    def encrypt(self, text: str, key: str) -> str:
        """
        Encrypt text using substitution cipher
        
        Args:
            text: Plain text to encrypt
            key: Substitution key (26 characters)
        
        Returns:
            Encrypted text
        """
        self.set_key(key)
        encrypted = ""
        for char in text.lower():
            if char.isalpha():
                encrypted += self.substitution_map.get(char, char)
            else:
                encrypted += char
        return encrypted
    
    def decrypt(self, text: str, key: str) -> str:
        """
        Decrypt text using substitution cipher
        
        Args:
            text: Encrypted text
            key: Substitution key (26 characters)
        
        Returns:
            Decrypted text
        """
        self.set_key(key)
        decrypted = ""
        for char in text.lower():
            if char.isalpha():
                decrypted += self.reverse_map.get(char, char)
            else:
                decrypted += char
        return decrypted

class DatasetGenerator:
    """Generate datasets for ML cryptanalysis"""
    
    def __init__(self):
        self.caesar = CaesarCipher()
        self.vigenere = VigenereCipher()
        self.substitution = SubstitutionCipher()
        
        # Sample texts for encryption
        self.sample_texts = [
            "the quick brown fox jumps over the lazy dog",
            "hello world this is a test message",
            "cryptography is the practice of secure communication",
            "machine learning can be applied to cryptanalysis",
            "frequency analysis is a powerful cryptanalytic technique",
            "the caesar cipher is one of the oldest encryption methods",
            "substitution ciphers replace each letter with another",
            "vigenere cipher uses a keyword for encryption",
            "breaking ciphers requires understanding patterns",
            "modern cryptography uses complex mathematical algorithms"
        ]
    
    def generate_caesar_dataset(self, num_samples: int = 1000) -> pd.DataFrame:
        """
        Generate dataset for Caesar cipher analysis
        
        Args:
            num_samples: Number of samples to generate
        
        Returns:
            DataFrame with encrypted text and labels
        """
        data = []
        
        for _ in range(num_samples):
            # Randomly select text and key
            text = random.choice(self.sample_texts)
            key = random.randint(1, 25)  # Avoid key=0 (no encryption)
            
            # Encrypt the text
            encrypted = self.caesar.encrypt(text, key)
            
            data.append({
                'original_text': text,
                'encrypted_text': encrypted,
                'key': key,
                'cipher_type': 'caesar'
            })
        
        return pd.DataFrame(data)
    
    def generate_vigenere_dataset(self, num_samples: int = 1000) -> pd.DataFrame:
        """
        Generate dataset for Vigenère cipher analysis
        
        Args:
            num_samples: Number of samples to generate
        
        Returns:
            DataFrame with encrypted text and labels
        """
        data = []
        
        # Common keywords for Vigenère
        keywords = ['key', 'secret', 'password', 'crypto', 'secure', 'hidden', 'code']
        
        for _ in range(num_samples):
            # Randomly select text and key
            text = random.choice(self.sample_texts)
            key = random.choice(keywords)
            
            # Encrypt the text
            encrypted = self.vigenere.encrypt(text, key)
            
            data.append({
                'original_text': text,
                'encrypted_text': encrypted,
                'key': key,
                'cipher_type': 'vigenere'
            })
        
        return pd.DataFrame(data)
    
    def generate_substitution_dataset(self, num_samples: int = 1000) -> pd.DataFrame:
        """
        Generate dataset for substitution cipher analysis
        
        Args:
            num_samples: Number of samples to generate
        
        Returns:
            DataFrame with encrypted text and labels
        """
        data = []
        
        for _ in range(num_samples):
            # Randomly select text and generate key
            text = random.choice(self.sample_texts)
            key = self.substitution.generate_key()
            
            # Encrypt the text
            encrypted = self.substitution.encrypt(text, key)
            
            data.append({
                'original_text': text,
                'encrypted_text': encrypted,
                'key': key,
                'cipher_type': 'substitution'
            })
        
        return pd.DataFrame(data)
    
    def generate_combined_dataset(self, samples_per_cipher: int = 500) -> pd.DataFrame:
        """
        Generate combined dataset with all cipher types
        
        Args:
            samples_per_cipher: Number of samples per cipher type
        
        Returns:
            DataFrame with all encrypted texts and labels
        """
        caesar_df = self.generate_caesar_dataset(samples_per_cipher)
        vigenere_df = self.generate_vigenere_dataset(samples_per_cipher)
        substitution_df = self.generate_substitution_dataset(samples_per_cipher)
        
        # Combine all datasets
        combined_df = pd.concat([caesar_df, vigenere_df, substitution_df], 
                              ignore_index=True)
        
        return combined_df
    
    def save_dataset(self, df: pd.DataFrame, filename: str):
        """
        Save dataset to CSV file
        
        Args:
            df: DataFrame to save
            filename: Output filename
        """
        df.to_csv(f"data/{filename}", index=False)
        logger.info(f"Dataset saved to data/{filename}")

def main():
    """Generate and save datasets"""
    generator = DatasetGenerator()
    
    # Create data directory if it doesn't exist
    import os
    os.makedirs("data", exist_ok=True)
    
    # Generate individual datasets
    logger.info("Generating Caesar cipher dataset...")
    caesar_df = generator.generate_caesar_dataset(1000)
    generator.save_dataset(caesar_df, "caesar_dataset.csv")
    
    logger.info("Generating Vigenère cipher dataset...")
    vigenere_df = generator.generate_vigenere_dataset(1000)
    generator.save_dataset(vigenere_df, "vigenere_dataset.csv")
    
    logger.info("Generating substitution cipher dataset...")
    substitution_df = generator.generate_substitution_dataset(1000)
    generator.save_dataset(substitution_df, "substitution_dataset.csv")
    
    logger.info("Generating combined dataset...")
    combined_df = generator.generate_combined_dataset(500)
    generator.save_dataset(combined_df, "combined_dataset.csv")
    
    logger.info("All datasets generated successfully!")

if __name__ == "__main__":
    main() 