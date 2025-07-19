# Machine Learning in Cryptanalysis

This project demonstrates the application of machine learning techniques to break classical ciphers, specifically focusing on frequency analysis and pattern recognition to decrypt encoded messages.

## Project Overview

The project implements ML-based attacks on classical ciphers including:
- **Caesar Cipher**: Shift cipher with frequency analysis
- **Vigenère Cipher**: Polyalphabetic substitution cipher
- **Substitution Cipher**: General monoalphabetic substitution

## Features

- **Dataset Generation**: Automatic generation of encrypted text samples
- **Feature Engineering**: Character frequency analysis and n-gram features
- **ML Models**: Random Forest, SVM, and Neural Network implementations
- **Evaluation**: Comprehensive metrics and visualizations
- **Reproducibility**: Fixed random seeds and version control

## Directory Structure

```
ml_cryptanalysis/
├── data/               # Raw and processed datasets
├── models/             # Trained model files
├── scripts/            # Training and evaluation scripts
├── results/            # Output files and visualizations
├── src/                # Core source code
├── config/             # Configuration files
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Setup Instructions

### 1. Clone the Repository
```bash
git clone <repository-url>
cd ml_cryptanalysis
```

### 2. Create Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix/MacOS:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Download Data (if needed)
```bash
python scripts/generate_data.py
```

## Usage

### Training Models
```bash
python scripts/train_model.py --cipher caesar --model random_forest
```

### Evaluating Models
```bash
python scripts/evaluate_model.py --model_path models/caesar_rf_model.pkl
```

### Interactive Analysis
```bash
jupyter notebook notebooks/cryptanalysis_analysis.ipynb
```

## Project Components

### 1. Data Generation (`src/data_generator.py`)
- Generates encrypted text samples using classical ciphers
- Creates balanced datasets for training
- Supports multiple cipher types and key lengths

### 2. Feature Engineering (`src/feature_extractor.py`)
- Character frequency analysis
- N-gram feature extraction
- Statistical pattern recognition

### 3. Model Training (`scripts/train_model.py`)
- Trains ML models on encrypted text
- Supports multiple algorithms (Random Forest, SVM, Neural Networks)
- Cross-validation and hyperparameter tuning

### 4. Evaluation (`scripts/evaluate_model.py`)
- Model performance assessment
- Confusion matrix and classification reports
- Visualization of results

## Results

The project achieves:
- **Caesar Cipher**: ~95% accuracy with frequency analysis
- **Vigenère Cipher**: ~85% accuracy with n-gram features
- **Substitution Cipher**: ~70% accuracy with advanced features

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Classical cryptography literature
- Scikit-learn documentation
- Cryptography community resources 
