# AG News Classification

A comprehensive text classification project implementing multiple machine learning approaches on the AG News dataset, including traditional ML baselines, CNN architectures, and transformer models.

## Overview

This project compares different text classification approaches:

- **Baseline**: Logistic Regression with TF-IDF features
- **TextCNN**: Convolutional Neural Network for text classification
- **DistilBERT**: Pre-trained transformer model fine-tuning

## Dataset

The AG News dataset contains news articles from 4 categories:
- World
- Sports  
- Business
- Science/Technology

## Installation & Setup

### Option 1: Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver.

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install dependencies
uv sync

# Activate the virtual environment
source .venv/bin/activate
```

### Option 2: Using pip

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate    # On Unix/macOS
# or
venv\Scripts\activate       # On Windows

# Install dependencies
pip install -r requirements.txt
```

**Note**: If `requirements.txt` doesn't exist, generate it from pyproject.toml:
```bash
pip install -e .
```

### Option 3: Using conda

```bash
# Create conda environment
conda create -n ag-news python=3.13
conda activate ag-news

# Install PyTorch (choose appropriate version for your system)
conda install pytorch torchvision torchaudio -c pytorch

# Install other dependencies
pip install -e .
```

## Usage

### 1. Exploratory Data Analysis

Generate comprehensive visualizations and statistics:

```bash
python eda.py
```

This creates visualizations in `outputs/eda/`:
- Overview dashboard
- Word clouds by category
- Statistical analysis
- Topic modeling results

### 2. Baseline Model (Logistic Regression)

Train the TF-IDF + Logistic Regression baseline:

```bash
python baseline_lr.py
```

Results saved to `outputs/baseline_lr/`

### 3. TextCNN Model

Train the Convolutional Neural Network:

```bash
# Fast mode (recommended): 8 trials, 6 epochs
uv run text_cnn.py fast

# Quick mode (testing): 1 trial, 3 epochs  
uv run text_cnn.py quick

# Full mode (thorough): 16 trials, 12 epochs
uv run text_cnn.py full

# Evaluate saved model
uv run text_cnn.py evaluate
```

Results saved to `outputs/textcnn/`

### 4. DistilBERT Model

Fine-tune the pre-trained transformer:

```bash
uv run distilbert_trainer.py
```

Results saved to `outputs/distilbert/`

## Model Performance

| Model | Test Accuracy | ROC AUC |
|-------|--------------|---------|
| Logistic Regression | 91.53% | 0.983 |
| TextCNN | 90.09% | 0.979 |
| DistilBERT | 94.70% | 0.993 |

## Requirements

- Python 3.13+
- PyTorch 2.8.0+
- Transformers 4.57.0+
- scikit-learn 1.7.2+
- See `pyproject.toml` for complete dependencies

## Hardware Requirements

- **CPU**: Multi-core recommended for data processing
- **GPU**: Optional but recommended for TextCNN and DistilBERT
  - CUDA-compatible GPU for NVIDIA
  - MPS support for Apple Silicon Macs
- **Memory**: 8GB+ RAM recommended
- **Storage**: ~2GB for datasets and model outputs

## Troubleshooting

### Common Issues

1. **MPS Device Errors (Apple Silicon)**:
   - The code includes automatic fallback to CPU if MPS fails
   - Set `PYTORCH_ENABLE_MPS_FALLBACK=1` if needed

2. **Memory Issues**:
   - Reduce batch size in training scripts
   - Use fewer workers for data loading

3. **Missing Data**:
   - Ensure `data/train.csv` and `data/test.csv` exist
   - Download AG News dataset if needed

### Performance Tips

- Use `uv` for fastest dependency installation
- Enable GPU acceleration when available
- Adjust `num_workers` based on your CPU cores
- Use fast mode for initial testing

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and ensure code quality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.