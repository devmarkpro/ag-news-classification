#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    roc_auc_score, 
    roc_curve
)
import warnings
import os
import multiprocessing as mp
import joblib
from itertools import cycle

warnings.filterwarnings('ignore')

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)

def get_num_workers():
    num_cores = mp.cpu_count()
    optimal_workers = min(8, max(1, num_cores - 1))
    print(f"System has {num_cores} CPU cores, using {optimal_workers} workers for sklearn")
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

def create_tfidf_features(
    X_train, 
    X_val, 
    X_test=None,
    ngram_range=(1, 2),
    min_df=2,
    max_features=200000
):
    print(f"Creating TF-IDF features with {max_features} max features...")
    
    tfidf = TfidfVectorizer(
        ngram_range=ngram_range,
        min_df=min_df,
        max_features=max_features
    )
    
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_val_tfidf = tfidf.transform(X_val)
    
    print(f"TF-IDF vocabulary size: {len(tfidf.vocabulary_)}")
    print(f"Training features shape: {X_train_tfidf.shape}")
    print(f"Validation features shape: {X_val_tfidf.shape}")
    
    result = {
        'X_train': X_train_tfidf,
        'X_val': X_val_tfidf,
        'vectorizer': tfidf
    }
    
    if X_test is not None:
        X_test_tfidf = tfidf.transform(X_test)
        result['X_test'] = X_test_tfidf
        print(f"Test features shape: {X_test_tfidf.shape}")
    
    return result

def train_logistic_regression(
    X_train, 
    y_train, 
    max_iter=2000, 
    n_jobs=None,
    random_state=42
):
    print("Training Logistic Regression classifier...")
    
    if n_jobs is None:
        n_jobs = get_num_workers()
    
    clf = LogisticRegression(
        max_iter=max_iter, 
        n_jobs=n_jobs,
        random_state=random_state
    )
    
    clf.fit(X_train, y_train)
    
    print("Training completed!")
    return clf

def evaluate_model(clf, X_val, y_val, class_names, dataset_name="Validation"):
    print(f"Evaluating model on {dataset_name} set...")
    
    pred = clf.predict(X_val)
    pred_proba = clf.predict_proba(X_val)
    
    accuracy = accuracy_score(y_val, pred)
    print(f"{dataset_name} Accuracy: {accuracy:.4f}")
    
    print(f"\n{dataset_name} Classification Report:")
    print(classification_report(y_val, pred, digits=4))
    
    roc_auc = roc_auc_score(y_val, pred_proba, multi_class='ovr')
    print(f"{dataset_name} ROC AUC: {roc_auc:.4f}")
    
    return {
        'accuracy': accuracy,
        'predictions': pred,
        'probabilities': pred_proba,
        'roc_auc': roc_auc
    }

def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    print("Generating confusion matrix...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues', 
        xticklabels=class_names, 
        yticklabels=class_names
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Logistic Regression')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    
    plt.show()

def plot_roc_curves(y_true, y_prob, class_names, save_path=None):
    print("Generating ROC curves...")
    
    roc_auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
    
    fpr = {}
    tpr = {}
    
    plt.figure(figsize=(8, 6))
    
    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_prob[:, i])
        plt.plot(fpr[i], tpr[i], label=f'{class_names[i]} (AUC={roc_auc_score(y_true == i, y_prob[:, i]):.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves - Logistic Regression (AUC = {roc_auc:.4f})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curves saved to {save_path}")
    
    plt.show()
    return roc_auc

def save_model_and_vectorizer(clf, vectorizer, model_path="outputs/baseline_lr/models/model.joblib", vectorizer_path="outputs/baseline_lr/models/vectorizer.joblib"):
    print("Saving model and vectorizer...")
    
    joblib.dump(clf, model_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    print(f"Model saved to: {model_path}")
    print(f"Vectorizer saved to: {vectorizer_path}")

def load_model_and_vectorizer(model_path="baseline_lr_model.joblib", vectorizer_path="tfidf_vectorizer.joblib"):
    print("Loading model and vectorizer...")
    
    clf = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    print("Model and vectorizer loaded successfully!")
    return clf, vectorizer

def run_hyperparameter_search(X_train, y_train, X_val, y_val):
    print("Running hyperparameter search...")
    
    param_grid = {
        'max_features': [50000, 100000, 200000],
        'ngram_range': [(1, 1), (1, 2), (1, 3)],
        'min_df': [1, 2, 5]
    }
    
    best_score = 0
    best_params = {}
    results = []
    
    for max_feat in param_grid['max_features']:
        for ngram in param_grid['ngram_range']:
            for min_df in param_grid['min_df']:
                print(f"Testing: max_features={max_feat}, ngram_range={ngram}, min_df={min_df}")
                
                # Create TF-IDF features
                tfidf_data = create_tfidf_features(
                    X_train, X_val,
                    ngram_range=ngram,
                    min_df=min_df,
                    max_features=max_feat
                )
                
                # Train model
                clf = train_logistic_regression(tfidf_data['X_train'], y_train)
                
                # Evaluate
                pred = clf.predict(tfidf_data['X_val'])
                score = accuracy_score(y_val, pred)
                
                results.append({
                    'max_features': max_feat,
                    'ngram_range': ngram,
                    'min_df': min_df,
                    'accuracy': score
                })
                
                if score > best_score:
                    best_score = score
                    best_params = {
                        'max_features': max_feat,
                        'ngram_range': ngram,
                        'min_df': min_df
                    }
                
                print(f"Accuracy: {score:.4f}")
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best accuracy: {best_score:.4f}")
    
    return best_params, results

def main():
    print("Starting Baseline Logistic Regression training for AG News classification")
    
    set_seed(42)
    
    df, test_df = load_and_prepare_data()
    
    class_names = ["World", "Sports", "Business", "Sci/Tech"]
    class_names_dict = {i+1: name for i, name in enumerate(class_names)}
    X_train, X_val, y_train, y_val = train_test_split(
        df["text"], 
        df["label"] - 1,  # Convert labels from 1-4 to 0-3
        test_size=0.2, 
        random_state=42, 
        stratify=df["label"]
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    tfidf_data = create_tfidf_features(
        X_train, X_val, test_df["text"],
        ngram_range=(1, 2),
        min_df=2,
        max_features=200000
    )
    
    clf = train_logistic_regression(tfidf_data['X_train'], y_train)
    
    val_results = evaluate_model(clf, tfidf_data['X_val'], y_val, class_names, "Validation")
    
    y_test = test_df["label"] - 1
    test_results = evaluate_model(clf, tfidf_data['X_test'], y_test, class_names, "Test")
    
    os.makedirs("outputs/baseline_lr/plots", exist_ok=True)
    os.makedirs("outputs/baseline_lr/models", exist_ok=True)
    print("\nGenerating evaluation plots...")
    
    plot_confusion_matrix(
        y_val, val_results['predictions'], class_names,
        save_path="outputs/baseline_lr/plots/confusion_matrix_val.png"
    )
    
    plot_confusion_matrix(
        y_test, test_results['predictions'], class_names,
        save_path="outputs/baseline_lr/plots/confusion_matrix_test.png"
    )
    
    # ROC curves for validation set
    val_roc_auc = plot_roc_curves(
        y_val, val_results['probabilities'], class_names,
        save_path="outputs/baseline_lr/plots/roc_curves_val.png"
    )
    
    # ROC curves for test set
    test_roc_auc = plot_roc_curves(
        y_test, test_results['probabilities'], class_names,
        save_path="outputs/baseline_lr/plots/roc_curves_test.png"
    )
    
    # Save model and vectorizer
    save_model_and_vectorizer(clf, tfidf_data['vectorizer'])
    
    # Print final summary
    print("\nBaseline training completed successfully!")
    print(f"Validation accuracy: {val_results['accuracy']:.4f}")
    print(f"Test accuracy: {test_results['accuracy']:.4f}")
    print(f"Validation ROC AUC: {val_results['roc_auc']:.4f}")
    print(f"Test ROC AUC: {test_results['roc_auc']:.4f}")
    print(f"Model saved as: outputs/baseline_lr/models/model.joblib")
    print(f"Vectorizer saved as: outputs/baseline_lr/models/vectorizer.joblib")
    print(f"Plots saved in: outputs/baseline_lr/plots/")
    
    return clf, tfidf_data['vectorizer'], val_results, test_results

if __name__ == "__main__":
    main()
