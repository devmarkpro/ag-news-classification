import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from collections import Counter
import random
import warnings
from sklearn.model_selection import train_test_split, ParameterSampler
import multiprocessing as mp
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def basic_tokenize(text: str):
    return text.lower().strip().replace("\n", " ").split()


def build_vocab(texts, min_freq=2, max_size=50000):
    counter = Counter()
    for t in texts:
        counter.update(basic_tokenize(t))
    stoi = {"<pad>": 0, "<unk>": 1}
    for tok, freq in counter.most_common():
        if freq < min_freq:
            break
        if len(stoi) >= max_size:
            break
        stoi[tok] = len(stoi)
    itos = {i: s for s, i in stoi.items()}
    return stoi, itos


def encode(text, stoi, max_len):
    toks = basic_tokenize(text)
    ids = [stoi.get(t, 1) for t in toks]
    if len(ids) >= max_len:
        return ids[:max_len]
    return ids + [0] * (max_len - len(ids))


class AGNewsDataset(Dataset):
    def __init__(self, df, stoi, max_len=200):
        self.texts = df["text"].tolist()
        self.labels = (df["label"].astype(int) - 1).tolist()
        self.stoi = stoi
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = torch.tensor(
            encode(self.texts[idx], self.stoi, self.max_len), dtype=torch.long
        )
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y


class TextCNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        emb_dim=200,
        num_classes=4,
        kernel_sizes=(3, 4, 5),
        num_channels=128,
        dropout=0.5,
        pad_idx=0,
    ):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList(
            [nn.Conv1d(emb_dim, num_channels, k) for k in kernel_sizes]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_channels * len(kernel_sizes), num_classes)

    def forward(self, x):
        e = self.emb(x).transpose(1, 2)
        h = [torch.relu(conv(e)).max(dim=2).values for conv in self.convs]
        h = torch.cat(h, dim=1)
        out = self.fc(self.dropout(h))
        return out


def accuracy(logits, labels):
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


def run_epoch(
    dataloader, model, criterion, optimizer=None, device="cpu", grad_clip=None
):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, total_acc, total_n = 0.0, 0.0, 0

    for x, y in dataloader:
        try:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        except Exception as e:
            print(f"Error moving tensors to device {device}: {e}")
            device = "cpu"
            x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_acc += accuracy(logits, y) * bs
        total_n += bs

    return total_loss / total_n, total_acc / total_n


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        try:
            test_tensor = torch.randn(10, 10).to("mps")
            test_result = test_tensor @ test_tensor.T
            return "mps"
        except Exception as e:
            print(f"MPS device test failed: {e}. Falling back to CPU.")
            return "cpu"
    else:
        return "cpu"


def get_num_workers():
    num_cores = mp.cpu_count()

    optimal_workers = min(4, max(1, num_cores // 2))

    print(f"System has {num_cores} CPU cores, using {optimal_workers} workers")
    return optimal_workers


def plot_confusion_matrix_textcnn(y_true, y_pred, class_names, save_path=None):
    print("Generating confusion matrix...")

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - TextCNN")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")

    plt.show()


def plot_roc_curves_textcnn(y_true, y_prob, class_names, save_path=None):
    print("Generating ROC curves...")

    roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr")

    fpr = {}
    tpr = {}

    plt.figure(figsize=(8, 6))

    for i in range(len(class_names)):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_prob[:, i])
        plt.plot(
            fpr[i],
            tpr[i],
            label=f"{class_names[i]} (AUC={roc_auc_score(y_true == i, y_prob[:, i]):.3f})",
        )

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves - TextCNN (AUC = {roc_auc:.4f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"ROC curves saved to {save_path}")

    plt.show()
    return roc_auc


def quick_search_mode():
    return {
        "emb_dim": [100],
        "channels": [128],
        "kernel_sizes": [(3, 4, 5)],
        "dropout": [0.3],
        "lr": [1e-3],
        "weight_decay": [1e-4],
        "batch_size": [128],
        "max_len": [200],
        "epochs": [3],
        "patience": [1],
        "grad_clip": [1.0],
        "optimizer": ["adamw"],
    }


def main(mode="fast"):
    print(f"Starting TextCNN training for AG News classification (mode: {mode})")

    set_seed(42)

    data_path = "data/"
    class_names = ["World", "Sports", "Business", "Sci/Tech"]
    class_names_dict = {i + 1: name for i, name in enumerate(class_names)}

    train_df = pd.read_csv(f"{data_path}/train.csv")
    test_df = pd.read_csv(f"{data_path}/test.csv")

    df = train_df.copy(deep=True)
    df["text"] = df["Title"].astype(str) + ". " + df["Description"].astype(str)
    df["label"] = df["Class Index"]
    df = df[["text", "label"]]

    test_df = test_df.copy(deep=True)
    test_df["text"] = (
        test_df["Title"].astype(str) + ". " + test_df["Description"].astype(str)
    )
    test_df["label"] = test_df["Class Index"]
    test_df = test_df[["text", "label"]]

    device = get_device()
    print("Using device:", device)

    # Get optimal number of workers
    num_workers = get_num_workers()

    # Build vocabulary
    train_texts = df["text"].tolist()
    stoi, itos = build_vocab(train_texts, min_freq=2, max_size=50000)
    vocab_size = len(stoi)
    print(f"Vocab size: {vocab_size}")

    # Split data
    train_df_, val_df_ = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )

    def train_eval_textcnn(cfg, train_df, val_df, trial_idx=0, seed=42, fast_mode=True):
        """
        cfg: dict with keys:
        emb_dim, channels, kernel_sizes (tuple), dropout, lr, weight_decay,
        batch_size, max_len, epochs, patience, grad_clip, optimizer ('adam'|'adamw')
        Returns dict with metrics and the path to the best checkpoint for this trial.
        """
        set_seed(seed + trial_idx)

        # --- Data ---
        ds_tr = AGNewsDataset(train_df, stoi, max_len=cfg["max_len"])
        ds_va = AGNewsDataset(val_df, stoi, max_len=cfg["max_len"])
        dl_tr = DataLoader(
            ds_tr,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=device != "cpu",  # Enable pin_memory for GPU devices
        )
        dl_va = DataLoader(
            ds_va,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device != "cpu",  # Enable pin_memory for GPU devices
        )

        # --- Model ---
        model = TextCNN(
            vocab_size=vocab_size,
            emb_dim=cfg["emb_dim"],
            num_classes=4,
            kernel_sizes=cfg["kernel_sizes"],
            num_channels=cfg["channels"],
            dropout=cfg["dropout"],
            pad_idx=0,
        ).to(device)

        # --- Optimizer ---
        if cfg["optimizer"].lower() == "adam":
            opt = torch.optim.Adam(
                model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
            )
        else:
            opt = torch.optim.AdamW(
                model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
            )

        crit = nn.CrossEntropyLoss()

        # --- Train with early stopping ---
        best_val_acc, best_epoch = -1.0, -1
        patience_left = cfg["patience"]
        ckpt_path = f"tune_trial{trial_idx:03d}_best.pt"

        try:
            for epoch in range(1, cfg["epochs"] + 1):
                # FIXED: Pass device parameter to run_epoch
                tr_loss, tr_acc = run_epoch(
                    dl_tr, model, crit, opt, device=device, grad_clip=cfg["grad_clip"]
                )
                va_loss, va_acc = run_epoch(dl_va, model, crit, device=device)

                # Only print every 2nd epoch in fast mode to reduce output
                if not fast_mode or epoch % 2 == 0 or epoch == cfg["epochs"]:
                    print(
                        f"[trial {trial_idx:02d}] ep {epoch:02d}/{cfg['epochs']}"
                        f" | train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f}"
                    )

                if va_acc > best_val_acc:
                    best_val_acc, best_epoch = va_acc, epoch
                    patience_left = cfg["patience"]
                    torch.save(model.state_dict(), ckpt_path)
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        if fast_mode:
                            print(
                                f"[trial {trial_idx:02d}] early stopping at epoch {epoch}"
                            )
                        break

        except RuntimeError as e:
            # Gracefully handle OOM or other runtime issues
            print(f"[trial {trial_idx:02d}] RuntimeError: {e}")
            # FIXED: Clear cache for both CUDA and MPS
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            return {
                "trial": trial_idx,
                "status": "error",
                "msg": str(e),
                "val_acc": np.nan,
                "val_loss": np.nan,
                "best_epoch": -1,
                "ckpt": None,
                "config": cfg,
            }

        return {
            "trial": trial_idx,
            "status": "ok",
            "msg": "",
            "val_acc": float(best_val_acc),
            "val_loss": float(va_loss),
            "best_epoch": best_epoch,
            "ckpt": ckpt_path,
            "config": cfg,
        }

    # Select hyperparameter search based on mode
    if mode == "quick":
        param_dist = quick_search_mode()
        N_TRIALS = 1
        print("Quick mode: 1 trial, 3 epochs (for testing)")
    elif mode == "fast":
        param_dist = {
            "emb_dim": [100, 200],  # Reduced from 3 to 2 options
            "channels": [128, 192],  # Reduced from 3 to 2 options
            "kernel_sizes": [(3, 4, 5)],  # Fixed to best performing option
            "dropout": [0.3, 0.5],  # Keep 2 options
            "lr": [1e-3, 2e-3],  # Reduced from 4 to 2 options
            "weight_decay": [0.0, 1e-4],  # Reduced from 3 to 2 options
            "batch_size": [128],  # Fixed to larger batch for speed
            "max_len": [200],  # Fixed to reasonable length
            "epochs": [6],  # Reduced from 12 to 6 epochs
            "patience": [2],  # Reduced patience for faster stopping
            "grad_clip": [1.0],  # Fixed to best practice
            "optimizer": ["adamw"],  # Fixed to best performing
        }
        N_TRIALS = 8
        print("Fast mode: 8 trials, 6 epochs (recommended)")
    else:  # full mode
        param_dist = {
            "emb_dim": [100, 200, 300],
            "channels": [128, 192, 256],
            "kernel_sizes": [(3, 4, 5), (2, 3, 4, 5)],
            "dropout": [0.3, 0.5],
            "lr": [5e-4, 1e-3, 2e-3, 3e-3],
            "weight_decay": [0.0, 1e-4, 5e-4],
            "batch_size": [64, 128],
            "max_len": [160, 200, 256],
            "epochs": [12],
            "patience": [3],
            "grad_clip": [0.0, 1.0],
            "optimizer": ["adamw", "adam"],
        }
        N_TRIALS = 16
        print("Full mode: 16 trials, 12 epochs (thorough search)")
    samples = list(ParameterSampler(param_dist, n_iter=N_TRIALS, random_state=42))

    results = []
    best = {"val_acc": -1.0}

    for i, cfg in enumerate(samples):
        print("\n" + "=" * 80)
        print(f"Trial {i+1}/{N_TRIALS} | config: {cfg}")
        out = train_eval_textcnn(
            cfg, train_df_, val_df_, trial_idx=i, seed=42, fast_mode=True
        )
        results.append(out)
        if out["status"] == "ok" and out["val_acc"] > best["val_acc"]:
            best = out
            # Save best model to standardized location
            os.makedirs("outputs/textcnn/models", exist_ok=True)
            torch.save(
                torch.load(best["ckpt"], map_location="cpu"),
                "outputs/textcnn/models/best_model.pt",
            )

    # Summarize results
    df_results = pd.DataFrame(
        [
            {
                "trial": r["trial"],
                "status": r["status"],
                "val_acc": r["val_acc"],
                "best_epoch": r["best_epoch"],
                **{f"cfg_{k}": v for k, v in r["config"].items()},
            }
            for r in results
        ]
    )

    df_results = df_results.sort_values("val_acc", ascending=False)
    # Create output directories
    os.makedirs("outputs/textcnn/plots", exist_ok=True)
    os.makedirs("outputs/textcnn/models", exist_ok=True)

    print("\n" + "=" * 80)
    print("RESULTS SUMMARY:")
    print(df_results.head(10))

    if best["val_acc"] > 0:
        print(f"\nTraining completed successfully!")
        print(f"Best trial: {best['trial']}")
        print(f"Best validation accuracy: {best['val_acc']:.4f}")
        print(f"Best epoch: {best['best_epoch']}")
        print(f"Best model saved as: outputs/textcnn/models/best_model.pt")
        print(f"Results saved in: outputs/textcnn/")

        # Evaluate on test set with best model
        print("\nEvaluating best model on test set...")
        best_model = TextCNN(
            vocab_size=vocab_size,
            emb_dim=best["config"]["emb_dim"],
            num_classes=4,
            kernel_sizes=best["config"]["kernel_sizes"],
            num_channels=best["config"]["channels"],
            dropout=best["config"]["dropout"],
            pad_idx=0,
        ).to(device)

        best_model.load_state_dict(
            torch.load("outputs/textcnn/models/best_model.pt", map_location=device)
        )

        # Create test dataset and dataloader
        ds_test = AGNewsDataset(test_df, stoi, max_len=best["config"]["max_len"])
        dl_test = DataLoader(
            ds_test,
            batch_size=best["config"]["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device != "cpu",
        )

        # Evaluate on test set
        crit = nn.CrossEntropyLoss()
        test_loss, test_acc = run_epoch(dl_test, best_model, crit, device=device)

        print(f"Test accuracy: {test_acc:.4f}")
        print(f"Test loss: {test_loss:.4f}")
    else:
        print("\nNo successful trials found. All trials encountered errors.")
        print("This suggests a fundamental issue with the model or data setup.")


def evaluate_saved_model():
    print("Starting TextCNN model evaluation on test set...")

    set_seed(42)

    data_path = "data/"
    class_names = ["World", "Sports", "Business", "Sci/Tech"]
    class_names_dict = {i + 1: name for i, name in enumerate(class_names)}

    print("Loading data...")
    df = pd.read_csv(f"{data_path}/train.csv")
    test_df = pd.read_csv(f"{data_path}/test.csv")

    # Prepare text data
    df["text"] = df["Title"].astype(str) + ". " + df["Description"].astype(str)
    test_df["text"] = (
        test_df["Title"].astype(str) + ". " + test_df["Description"].astype(str)
    )

    # Build vocabulary from training data
    train_texts = df["text"].tolist()
    stoi, itos = build_vocab(train_texts, min_freq=2, max_size=50000)
    vocab_size = len(stoi)
    print(f"Vocab size: {vocab_size}")

    # Check if model exists
    model_path = "outputs/textcnn/models/best_model.pt"
    # Also check for the alternative name with underscore
    if not os.path.exists(model_path):
        model_path = "outputs/textcnn/models/best_model_.pt"
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please run training first with: python text_cnn.py fast")
        return

    config = {
        "emb_dim": 100,
        "channels": 128,
        "kernel_sizes": (3, 4, 5),
        "dropout": 0.3,
        "max_len": 200,
    }

    print(f"📦 Loading model from {model_path}...")

    test_df_copy = test_df.copy()
    test_df_copy["label"] = test_df_copy["Class Index"]
    
    test_labels = test_df["Class Index"].values - 1  # Convert to 0-indexed
    ds_test = AGNewsDataset(test_df_copy, stoi, max_len=config["max_len"])
    dl_test = DataLoader(ds_test, batch_size=128, shuffle=False)

    device = get_device()
    model = TextCNN(
        vocab_size=vocab_size,
        emb_dim=config["emb_dim"],
        num_channels=config["channels"],
        kernel_sizes=config["kernel_sizes"],
        dropout=config["dropout"],
        num_classes=4,
    ).to(device)

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []

    print("Evaluating on test set...")
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dl_test):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    y_true = np.array(all_labels)
    y_pred = np.array(all_predictions)
    y_prob = np.array(all_probabilities)

    accuracy = np.mean(y_true == y_pred)
    print(f"Test Accuracy: {accuracy:.4f}")

    os.makedirs("outputs/textcnn/plots", exist_ok=True)

    print("\nGenerating evaluation plots...")

    plot_confusion_matrix_textcnn(
        y_true,
        y_pred,
        class_names,
        save_path="outputs/textcnn/plots/confusion_matrix_test.png",
    )

    test_roc_auc = plot_roc_curves_textcnn(
        y_true,
        y_prob,
        class_names,
        save_path="outputs/textcnn/plots/roc_curves_test.png",
    )

    print("\nTextCNN evaluation completed successfully!")
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Test ROC AUC: {test_roc_auc:.4f}")
    print(f"Plots saved in: outputs/textcnn/plots/")

    return accuracy, test_roc_auc


if __name__ == "__main__":
    import sys

    mode = "fast"
    if len(sys.argv) > 1:
        mode = sys.argv[1].lower()
        if mode not in ["quick", "fast", "full", "evaluate"]:
            print("Invalid mode. Use: quick, fast, full, or evaluate")
            print("Usage: python text_cnn.py [quick|fast|full|evaluate]")
            print("   quick:    1 trial, 3 epochs (~2 minutes)")
            print("   fast:     8 trials, 6 epochs (~15 minutes) [default]")
            print("   full:     16 trials, 12 epochs (~60+ minutes)")
            print("   evaluate: Load saved model and evaluate on test set")
            sys.exit(1)

    if mode == "evaluate":
        evaluate_saved_model()
    else:
        main(mode)
