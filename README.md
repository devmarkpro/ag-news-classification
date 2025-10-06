# AG News Classification

A comprehensive text classification project implementing multiple machine learning approaches on the AG News dataset, including traditional ML baselines, CNN architectures, and transformer models.

## Overview

This project compares different text classification approaches:

- **Baseline**: Logistic Regression with TF-IDF features
- **TextCNN**: Convolutional Neural Network with Optuna hyperparameter optimization
- **DistilBERT**: Pre-trained transformer model fine-tuning

## Key Features

- **Advanced Hyperparameter Optimization**: Optuna integration with MedianPruner for intelligent trial pruning
- **Comprehensive Experiment Tracking**: Weights & Biases (wandb) integration for real-time monitoring
- **Intelligent Parameter Search**: Optuna's default TPE-based optimization for efficient hyperparameter exploration
- **Professional Visualizations**: Confusion matrices, ROC curves, and parameter importance plots

## Dataset

The [AG News dataset](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset) contains news articles from 4 categories:
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

### 3. TextCNN Model with Optuna Optimization

Train the Convolutional Neural Network with intelligent hyperparameter optimization:

```bash
# Default: 20 trials with MedianPruner optimization
python text_cnn.py

# Custom number of trials
python text_cnn.py 10
```

**Optuna Features:**
- **Intelligent Search**: 10+ hyperparameters optimized simultaneously
- **Early Stopping**: MedianPruner stops unpromising trials (typically saves 60-70% compute time)
- **Advanced Sampling**: TPE algorithm for efficient parameter space exploration
- **Comprehensive Tracking**: Full wandb integration with trial-level metrics

**Hyperparameters Optimized:**
- Embedding dimensions: [100, 200, 300]
- Convolutional channels: [128, 192, 256]
- Kernel sizes: [(3,4,5), (2,3,4,5), (3,4,5,6)]
- Dropout rates: 0.1-0.6
- Learning rates: 1e-4 to 5e-3 (log scale)
- Batch sizes: [64, 128, 256]
- Weight decay, gradient clipping, optimizers

Results saved to `outputs/textcnn/`:
- Best model: `models/best_model.pt`
- Trial results: `optuna_trials.csv`
- Visualizations: confusion matrices, ROC curves

### 4. DistilBERT Model

Fine-tune the pre-trained transformer:

```bash
python distilbert_trainer.py
```

Results saved to `outputs/distilbert/`

## Weights & Biases Integration

All models include comprehensive experiment tracking with wandb:

### Setup wandb (Optional but Recommended)

```bash
# Install wandb
pip install wandb

# Login to your account
wandb login
```

### Features Tracked:
- **Real-time Metrics**: Loss, accuracy, learning rate per batch/epoch
- **Hyperparameter Logging**: Complete configuration for reproducibility
- **Model Artifacts**: Saved model checkpoints with versioning
- **Visualizations**: Confusion matrices, ROC curves automatically uploaded
- **Trial Comparison**: Side-by-side comparison of different hyperparameter combinations
- **Parameter Importance**: Optuna integration shows which parameters matter most

### View Results:
Visit [wandb.ai](https://wandb.ai) to view your experiment dashboard with:
- Training curves and metrics
- Hyperparameter sweep visualizations
- Model performance comparisons
- Parameter importance plots

## Model Performance

| Model               | Test Accuracy | Validation Accuracy | ROC AUC | Notes                        |
| ------------------- | ------------- | ------------------- | ------- | ---------------------------- |
| Logistic Regression | 91.53%        | 91.88%              | 0.983   | Fast, interpretable baseline |
| TextCNN (Optuna)    | 84.43%        | 92.31%              | 0.982   | Shows generalization gap     |
| DistilBERT          | 94.70%        | 94.94%              | 0.993   | Best overall performance     |

**Key Insights:**
- **DistilBERT** achieves the highest accuracy with excellent generalization
- **Logistic Regression** surprisingly outperforms TextCNN on test data despite being simpler
- **TextCNN with Optuna** demonstrates the importance of proper validation - high validation accuracy (92.31%) but lower test performance (84.43%) reveals overfitting challenges

## Requirements

- Python 3.13+
- PyTorch 2.8.0+
- Transformers 4.57.0+
- scikit-learn 1.7.2+
- **optuna**: For hyperparameter optimization
- **wandb**: For experiment tracking (optional but recommended)
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
- **For TextCNN optimization**: Start with fewer trials (5-10) for initial testing
- **wandb Integration**: Set up wandb for better experiment tracking and visualization
- **Optuna Pruning**: Let MedianPruner stop bad trials early to save compute time
