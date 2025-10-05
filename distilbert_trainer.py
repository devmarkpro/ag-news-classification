#!/usr/bin/env python3

import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, auc
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer
)
import evaluate
from itertools import cycle
import warnings
import os
import multiprocessing as mp

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

def get_num_workers():
    num_cores = mp.cpu_count()
    optimal_workers = min(4, max(1, num_cores // 2))
    
    print(f"System has {num_cores} CPU cores, using {optimal_workers} workers for data processing")
    return optimal_workers

def load_and_prepare_data(data_path="data/"):
    print("Loading data...")
    
    train_df = pd.read_csv(f"{data_path}/train.csv")
    test_df = pd.read_csv(f"{data_path}/test.csv")
    
    df = train_df.copy(deep=True)
    df['text'] = df['Title'].astype(str) + ". " + df['Description'].astype(str)
    df['label'] = df['Class Index']
    df = df[['text', 'label']]
    
    test_df = test_df.copy(deep=True)
    test_df['text'] = test_df['Title'].astype(str) + ". " + test_df['Description'].astype(str)
    test_df['label'] = test_df['Class Index']
    test_df = test_df[['text', 'label']]
    
    print(f"Training data shape: {df.shape}")
    print(f"Test data shape: {test_df.shape}")
    
    return df, test_df

def prepare_transformer_data(df, test_size=0.2, random_state=42):
    df = df.copy(deep=True)
    
    if df["label"].min() == 1:
        df = df.assign(label=df["label"] - 1)
    
    train_df, val_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["label"]
    )
    
    num_labels = int(df["label"].nunique())
    class_names = ["World", "Sports", "Business", "Sci/Tech"][:num_labels]
    class_names_dict = {i+1: name for i, name in enumerate(class_names)}
    
    print(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
    print(f"Number of classes: {num_labels}")
    print(f"Class names: {class_names}")
    
    return train_df, val_df, num_labels, class_names

def create_datasets_and_tokenize(train_df, val_df, model_name="distilbert-base-uncased", max_length=256, num_workers=None):
    print(f"Creating datasets and tokenizing with {model_name}...")
    
    if num_workers is None:
        num_workers = get_num_workers()
    
    train_ds = Dataset.from_pandas(train_df[["text", "label"]].reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df[["text", "label"]].reset_index(drop=True))
    ds = DatasetDict({"train": train_ds, "val": val_ds})
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    def tokenize(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=max_length
        )
    
    print(f"Tokenizing datasets using {num_workers} workers...")
    tokenized_ds = ds.map(tokenize, batched=True, num_proc=num_workers)
    tokenized_ds = tokenized_ds.remove_columns(["text"])
    tokenized_ds = tokenized_ds.rename_column("label", "labels")
    tokenized_ds.set_format("torch")
    
    print("Tokenization complete!")
    return tokenized_ds, tokenizer

def setup_metrics():
    acc_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        return {
            "accuracy": acc_metric.compute(predictions=preds, references=labels)["accuracy"],
            "macro_f1": f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"],
        }
    
    return compute_metrics

def train_distilbert(
    tokenized_ds, 
    tokenizer, 
    num_labels, 
    compute_metrics,
    model_name="distilbert-base-uncased",
    output_dir="outputs/distilbert/models",
    learning_rate=2e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    weight_decay=0.01,
    seed=42,
    dataloader_num_workers=None
):
    """Train DistilBERT model"""
    print("Setting up model and training arguments...")
    
    # Get optimal number of workers if not specified
    if dataloader_num_workers is None:
        dataloader_num_workers = get_num_workers()
    
    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    
    # Training arguments
    args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=weight_decay,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        logging_steps=50,
        seed=seed,
        fp16=torch.cuda.is_available(),  # mixed precision on CUDA
        dataloader_num_workers=dataloader_num_workers,  # Enable multiprocessing for data loading
    )
    
    print(f"Using {dataloader_num_workers} workers for data loading during training")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["val"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    
    print("Starting training...")
    trainer.train()
    
    print("Evaluating model...")
    metrics = trainer.evaluate()
    print("Final metrics:", metrics)
    
    return trainer, model, metrics

def plot_confusion_matrix(trainer, tokenized_ds, class_names, save_path=None):
    """Plot confusion matrix for validation set"""
    print("Generating confusion matrix...")
    
    # Get predictions
    pred_out = trainer.predict(tokenized_ds["val"])
    val_logits = pred_out.predictions
    val_labels = pred_out.label_ids
    val_preds = val_logits.argmax(axis=1)
    
    # Create confusion matrix
    cm = confusion_matrix(val_labels, val_preds, labels=list(range(len(class_names))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    fig = plt.figure(figsize=(7, 6))
    disp.plot(values_format="d", cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix — DistilBERT (Validation)")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()
    return val_logits, val_labels, val_preds

def plot_roc_curves(val_logits, val_labels, class_names, save_path=None):
    """Plot ROC curves for all classes"""
    print("Generating ROC curves...")
    
    # Convert logits to probabilities
    probs = torch.softmax(torch.tensor(val_logits), dim=1).numpy()  # [N, C]
    Y = np.eye(len(class_names))[val_labels]  # binarize labels
    
    # Calculate AUC scores
    auc_macro = roc_auc_score(Y, probs, average="macro", multi_class="ovr")
    auc_micro = roc_auc_score(Y, probs, average="micro", multi_class="ovr")
    print(f"ROC AUC — macro: {auc_macro:.4f} | micro: {auc_micro:.4f}")
    
    # Calculate ROC curves for each class
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(Y[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    plt.figure(figsize=(8, 6))
    for i, _ in zip(range(len(class_names)), cycle([None] * len(class_names))):
        plt.plot(fpr[i], tpr[i], lw=2, label=f"{class_names[i]} (AUC={roc_auc[i]:.3f})")
    
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves (OvR) — DistilBERT\nmacro AUC={auc_macro:.3f} | micro AUC={auc_micro:.3f}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")
    
    plt.show()
    return auc_macro, auc_micro

def evaluate_on_test_set(trainer, test_df, tokenizer, class_names, max_length=256):
    """Evaluate the trained model on test set"""
    print("Evaluating on test set...")
    
    # Prepare test data
    test_df_processed = test_df.copy()
    if test_df_processed["label"].min() == 1:
        test_df_processed = test_df_processed.assign(label=test_df_processed["label"] - 1)
    
    # Create test dataset
    from datasets import Dataset
    test_ds = Dataset.from_pandas(test_df_processed[["text", "label"]].reset_index(drop=True))
    
    def tokenize(examples):
        return tokenizer(
            examples["text"], 
            truncation=True, 
            padding="max_length", 
            max_length=max_length
        )
    
    # Tokenize test dataset
    test_ds = test_ds.map(tokenize, batched=True, num_proc=get_num_workers())
    test_ds = test_ds.remove_columns(["text"])
    test_ds = test_ds.rename_column("label", "labels")
    test_ds.set_format("torch")
    
    # Evaluate
    test_results = trainer.evaluate(test_ds)
    print(f"Test Results: {test_results}")
    
    # Get predictions for detailed analysis
    pred_out = trainer.predict(test_ds)
    test_logits = pred_out.predictions
    test_labels = pred_out.label_ids
    test_preds = test_logits.argmax(axis=1)
    
    return test_results, test_logits, test_labels, test_preds

def plot_confusion_matrix_from_data(y_true, y_pred, class_names, save_path=None, title="Confusion Matrix"):
    """Plot confusion matrix from data arrays"""
    print("Generating confusion matrix...")
    
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    fig = plt.figure(figsize=(7, 6))
    disp.plot(values_format="d", cmap="Blues", colorbar=False)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()

def plot_roc_curves_from_data(logits, labels, class_names, save_path=None, title="ROC Curves"):
    """Plot ROC curves from logits and labels"""
    print("Generating ROC curves...")
    
    # Convert logits to probabilities
    import torch
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()  # [N, C]
    Y = np.eye(len(class_names))[labels]  # binarize labels
    
    # Calculate AUC scores
    auc_macro = roc_auc_score(Y, probs, average="macro", multi_class="ovr")
    auc_micro = roc_auc_score(Y, probs, average="micro", multi_class="ovr")
    print(f"ROC AUC — macro: {auc_macro:.4f} | micro: {auc_micro:.4f}")
    
    # Calculate ROC curves for each class
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(Y[:, i], probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Plot ROC curves
    plt.figure(figsize=(8, 6))
    for i, _ in zip(range(len(class_names)), cycle([None] * len(class_names))):
        plt.plot(fpr[i], tpr[i], lw=2, label=f"{class_names[i]} (AUC={roc_auc[i]:.3f})")
    
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"{title}\nmacro AUC={auc_macro:.3f} | micro AUC={auc_micro:.3f}")
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")
    
    plt.show()
    return auc_macro, auc_micro

def main():
    """Main training pipeline"""
    print("Starting DistilBERT training for AG News classification")
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Load and prepare data
    df, test_df = load_and_prepare_data()
    train_df, val_df, num_labels, class_names = prepare_transformer_data(df)
    
    # Create datasets and tokenize
    tokenized_ds, tokenizer = create_datasets_and_tokenize(train_df, val_df)
    
    # Setup metrics
    compute_metrics = setup_metrics()
    
    # Train model
    trainer, model, metrics = train_distilbert(
        tokenized_ds=tokenized_ds,
        tokenizer=tokenizer,
        num_labels=num_labels,
        compute_metrics=compute_metrics
    )
    
    # Generate visualizations
    print("\nGenerating evaluation plots...")
    
    # Create output directories
    os.makedirs("outputs/distilbert/plots", exist_ok=True)
    os.makedirs("outputs/distilbert/models", exist_ok=True)
    
    # Plot confusion matrix
    val_logits, val_labels, val_preds = plot_confusion_matrix(
        trainer, tokenized_ds, class_names, 
        save_path="outputs/distilbert/plots/confusion_matrix_val.png"
    )
    
    # Plot ROC curves
    auc_macro, auc_micro = plot_roc_curves(
        val_logits, val_labels, class_names,
        save_path="outputs/distilbert/plots/roc_curves_val.png"
    )
    
    # Evaluate on test set
    test_results, test_logits, test_labels, test_preds = evaluate_on_test_set(
        trainer, test_df, tokenizer, class_names
    )
    
    # Plot test set visualizations
    plot_confusion_matrix_from_data(
        test_labels, test_preds, class_names,
        save_path="outputs/distilbert/plots/confusion_matrix_test.png",
        title="Confusion Matrix — DistilBERT (Test)"
    )
    
    test_auc_macro, test_auc_micro = plot_roc_curves_from_data(
        test_logits, test_labels, class_names,
        save_path="outputs/distilbert/plots/roc_curves_test.png",
        title="ROC Curves (OvR) — DistilBERT (Test)"
    )
    
    # Print final summary
    print("\nTraining completed successfully!")
    print(f"Validation accuracy: {metrics.get('eval_accuracy', 'N/A'):.4f}")
    print(f"Test accuracy: {test_results.get('eval_accuracy', 'N/A'):.4f}")
    print(f"Validation F1 (macro): {metrics.get('eval_macro_f1', 'N/A'):.4f}")
    print(f"Test F1 (macro): {test_results.get('eval_macro_f1', 'N/A'):.4f}")
    print(f"Validation ROC AUC (macro): {auc_macro:.4f}")
    print(f"Test ROC AUC (macro): {test_auc_macro:.4f}")
    print(f"Model saved in: outputs/distilbert/models/")
    print(f"Plots saved in: outputs/distilbert/plots/")

if __name__ == "__main__":
    main()
