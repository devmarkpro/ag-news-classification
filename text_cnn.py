import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn
from collections import Counter
import random
import warnings
from sklearn.model_selection import train_test_split
import multiprocessing as mp
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
from torchtext.data.utils import get_tokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
import wandb
import optuna
from optuna.integration.wandb import WeightsAndBiasesCallback

warnings.filterwarnings("ignore")
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("wordnet", quiet=True)


tokenizer = get_tokenizer("basic_english")
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def build_vocab(texts, min_freq=2, max_size=50000):
    counter = Counter()
    for t in texts:
        counter.update(tokenizer(t))
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
    toks = tokenizer(text)
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
    dataloader,
    model,
    criterion,
    optimizer=None,
    scheduler=None,
    device="cpu",
    grad_clip=None,
    epoch=None,
    log_interval=50,
    verbose=True,
    log_wandb=True,
):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, total_acc, total_n = 0.0, 0.0, 0

    # Initialize logging variables
    batch_losses = []
    batch_accs = []
    learning_rates = []

    mode = "Train" if is_train else "Val"
    total_batches = len(dataloader)

    if verbose and epoch is not None:
        print(f"\n{mode} Epoch {epoch} - {total_batches} batches")
        print("-" * 50)

    for batch_idx, (x, y) in enumerate(dataloader):
        try:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        except Exception as e:
            print(f"Error moving tensors to device {device}: {e}")
            device = "cpu"
            x, y = x.to(device), y.to(device)

        logits = model(x)
        loss = criterion(logits, y)
        batch_acc = accuracy(logits, y)

        if is_train:
            # Get current learning rate before optimizer step
            current_lr = optimizer.param_groups[0]["lr"]
            learning_rates.append(current_lr)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if grad_clip:
                grad_norm = nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

        bs = x.size(0)
        batch_loss = loss.item()
        batch_losses.append(batch_loss)
        batch_accs.append(batch_acc)

        total_loss += batch_loss * bs
        total_acc += batch_acc * bs
        total_n += bs

        # Log batch progress
        if verbose and is_train and (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / total_n
            avg_acc = total_acc / total_n
            current_lr = optimizer.param_groups[0]["lr"] if optimizer else 0
            progress = (batch_idx + 1) / total_batches * 100

            print(
                f"  Batch {batch_idx + 1:4d}/{total_batches} ({progress:5.1f}%) | "
                f"Loss: {batch_loss:.4f} (avg: {avg_loss:.4f}) | "
                f"Acc: {batch_acc:.4f} (avg: {avg_acc:.4f}) | "
                f"LR: {current_lr:.2e}"
            )

            # Log to wandb
            if log_wandb and wandb.run is not None:
                wandb.log(
                    {
                        f"{mode.lower()}_batch_loss": batch_loss,
                        f"{mode.lower()}_batch_acc": batch_acc,
                        f"{mode.lower()}_avg_loss": avg_loss,
                        f"{mode.lower()}_avg_acc": avg_acc,
                        "learning_rate": current_lr,
                        "batch": batch_idx + 1,
                        "epoch": epoch if epoch is not None else 0,
                    }
                )

    # Calculate final metrics
    final_loss = total_loss / total_n
    final_acc = total_acc / total_n

    # Log epoch summary
    if verbose:
        if is_train and learning_rates:
            final_lr = learning_rates[-1]
            print(f"\n{mode} Summary:")
            print(f"  Final Loss: {final_loss:.4f} | Final Acc: {final_acc:.4f}")
            print(f"  Final LR: {final_lr:.2e}")
            print(
                f"  Loss std: {np.std(batch_losses):.4f} | Acc std: {np.std(batch_accs):.4f}"
            )
        else:
            print(f"\n{mode} Summary:")
            print(f"  Final Loss: {final_loss:.4f} | Final Acc: {final_acc:.4f}")
            print(
                f"  Loss std: {np.std(batch_losses):.4f} | Acc std: {np.std(batch_accs):.4f}"
            )

    # Log epoch summary to wandb
    if log_wandb and wandb.run is not None:
        log_dict = {
            f"{mode.lower()}_epoch_loss": final_loss,
            f"{mode.lower()}_epoch_acc": final_acc,
            f"{mode.lower()}_loss_std": np.std(batch_losses),
            f"{mode.lower()}_acc_std": np.std(batch_accs),
            "epoch": epoch if epoch is not None else 0,
        }

        if is_train and learning_rates:
            log_dict["final_learning_rate"] = learning_rates[-1]

        wandb.log(log_dict)

    return final_loss, final_acc


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


def preprocess_data(df):
    df["text"] = df["Title"].astype(str) + ". " + df["Description"].astype(str)
    df["label"] = df["Class Index"]
    df = df[["text", "label"]]

    # convert to lowercase
    df["text"] = df["text"].str.lower()

    # remove extra whitespace
    df["text"] = df["text"].str.replace(r"\s+", " ", regex=True)

    # remove new lines
    df["text"] = df["text"].str.replace(r"\n", "", regex=True)

    df["text"] = df["text"].apply(
        lambda x: " ".join([word for word in x.split() if word not in stop_words])
    )

    df["text"] = df["text"].apply(
        lambda x: " ".join([lemmatizer.lemmatize(word) for word in x.split()])
    )

    # remove special characters
    df["text"] = df["text"].str.replace(r"[^a-zA-Z0-9 ]", "", regex=True)

    # remove single characters
    df["text"] = df["text"].str.replace(r"\b\w\b", "", regex=True)

    return df


def main(n_trials=20):
    print(
        f"Starting TextCNN hyperparameter optimization with Optuna ({n_trials} trials)"
    )

    set_seed(42)

    data_path = "data/"

    df = pd.read_csv(f"{data_path}/train.csv")
    test_df = pd.read_csv(f"{data_path}/test.csv")

    df = preprocess_data(df)

    test_df["text"] = (
        test_df["Title"].astype(str) + ". " + test_df["Description"].astype(str)
    ).str.lower()
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

    def objective(trial):
        """
        Optuna objective function for hyperparameter optimization.
        """
        # Suggest hyperparameters
        cfg = {
            "emb_dim": trial.suggest_categorical("emb_dim", [100, 200, 300]),
            "channels": trial.suggest_categorical("channels", [128, 192, 256]),
            "kernel_sizes": trial.suggest_categorical(
                "kernel_sizes", [(3, 4, 5), (2, 3, 4, 5), (3, 4, 5, 6)]
            ),
            "dropout": trial.suggest_float("dropout", 0.1, 0.6, step=0.1),
            "lr": trial.suggest_float("lr", 1e-4, 5e-3, log=True),
            "weight_decay": trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True),
            "batch_size": trial.suggest_categorical("batch_size", [64, 128, 256]),
            "max_len": trial.suggest_categorical("max_len", [160, 200, 256]),
            "epochs": 20,
            "patience": 3,
            "grad_clip": trial.suggest_float("grad_clip", 0.5, 2.0, step=0.5),
            "optimizer": trial.suggest_categorical("optimizer", ["adam", "adamw"]),
        }

        # Initialize wandb for this trial
        wandb.init(
            project="ag-news-textcnn-optuna",
            name=f"trial_{trial.number:03d}",
            config={
                **cfg,
                "trial_number": trial.number,
                "vocab_size": vocab_size,
                "device": device,
            },
            reinit=True,
            tags=["textcnn", "ag-news", "optuna"],
        )

        try:
            result = train_textcnn_single(cfg, train_df_, val_df_, trial)
            wandb.finish()
            return result["val_acc"]
        except Exception as e:
            print(f"Trial {trial.number} failed: {e}")
            wandb.log({"error": str(e), "status": "failed"})
            wandb.finish()
            raise optuna.TrialPruned()

    def train_textcnn_single(cfg, train_df, val_df, trial=None):
        ds_tr = AGNewsDataset(train_df, stoi, max_len=cfg["max_len"])
        ds_va = AGNewsDataset(val_df, stoi, max_len=cfg["max_len"])
        dl_tr = DataLoader(
            ds_tr,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=num_workers,
            pin_memory=device != "cuda",
        )
        dl_va = DataLoader(
            ds_va,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device != "cuda",
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

        # Log model architecture to wandb
        wandb.watch(model, log="all", log_freq=100)

        # --- Optimizer ---
        if cfg["optimizer"].lower() == "adam":
            opt = torch.optim.Adam(
                model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
            )
        else:
            opt = torch.optim.AdamW(
                model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
            )
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=cfg["lr"], steps_per_epoch=len(dl_tr), epochs=cfg["epochs"]
        )
        crit = nn.CrossEntropyLoss(label_smoothing=0.05)

        # --- Train with early stopping ---
        best_val_acc, best_epoch = -1.0, -1
        patience_left = cfg["patience"]
        ckpt_path = "best_model.pt"

        try:
            for epoch in range(1, cfg["epochs"] + 1):
                tr_loss, tr_acc = run_epoch(
                    dl_tr,
                    model,
                    crit,
                    opt,
                    sched,
                    device=device,
                    grad_clip=cfg["grad_clip"],
                    epoch=epoch,
                    log_interval=50,
                    verbose=True,
                )
                va_loss, va_acc = run_epoch(
                    dl_va, model, crit, device=device, epoch=epoch, verbose=True
                )

                # Print epoch results
                print(
                    f"Epoch {epoch:02d}/{cfg['epochs']} | "
                    f"train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f}"
                )

                # Log epoch summary to wandb
                wandb.log(
                    {
                        "train_loss": tr_loss,
                        "train_acc": tr_acc,
                        "val_loss": va_loss,
                        "val_acc": va_acc,
                        "epoch": epoch,
                    }
                )

                # Report intermediate value to Optuna for pruning
                if trial is not None:
                    trial.report(va_acc, epoch)
                    if trial.should_prune():
                        print(f"Trial pruned at epoch {epoch}")
                        raise optuna.TrialPruned()

                if va_acc > best_val_acc:
                    best_val_acc, best_epoch = va_acc, epoch
                    patience_left = cfg["patience"]
                    torch.save(model.state_dict(), ckpt_path)

                    # Log best metrics to wandb
                    wandb.log(
                        {
                            "best_val_acc": best_val_acc,
                            "best_epoch": best_epoch,
                        }
                    )
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        print(f"Early stopping at epoch {epoch}")
                        break

        except RuntimeError as e:
            # Gracefully handle OOM or other runtime issues
            print(f"RuntimeError: {e}")

            # Log error to wandb
            wandb.log({"error": str(e), "status": "failed"})

            # Clear cache for both CUDA and MPS
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            return {
                "status": "error",
                "msg": str(e),
                "val_acc": np.nan,
                "val_loss": np.nan,
                "best_epoch": -1,
                "ckpt": None,
            }

        # Log final results to wandb
        wandb.log(
            {
                "final_best_val_acc": float(best_val_acc),
                "final_val_loss": float(va_loss),
                "final_best_epoch": best_epoch,
                "status": "completed",
            }
        )

        # Save model as wandb artifact
        if os.path.exists(ckpt_path):
            artifact = wandb.Artifact("best_model", type="model")
            artifact.add_file(ckpt_path)
            wandb.log_artifact(artifact)

        return {
            "status": "ok",
            "msg": "",
            "val_acc": float(best_val_acc),
            "val_loss": float(va_loss),
            "best_epoch": best_epoch,
            "ckpt": ckpt_path,
        }

    # Create Optuna study
    print("\n" + "=" * 80)
    print("Starting Optuna hyperparameter optimization...")

    # Initialize main wandb run for study tracking
    wandb.init(
        project="ag-news-textcnn-optuna",
        name="optuna_study",
        tags=["textcnn", "ag-news", "optuna", "study"],
    )

    # Create study with pruning
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=3, interval_steps=1
        ),
        study_name="textcnn_optimization",
    )

    # Add wandb callback for study tracking
    wandb_callback = WeightsAndBiasesCallback(
        metric_name="val_acc", wandb_kwargs={"project": "ag-news-textcnn-optuna"}
    )

    # Optimize
    study.optimize(objective, n_trials=n_trials, callbacks=[wandb_callback])

    # Get best trial
    best_trial = study.best_trial
    print(f"\nOptimization completed!")
    print(f"Best trial: {best_trial.number}")
    print(f"Best validation accuracy: {best_trial.value:.4f}")
    print(f"Best parameters: {best_trial.params}")

    # Log study results to main wandb run
    wandb.log(
        {
            "best_trial_number": best_trial.number,
            "best_val_accuracy": best_trial.value,
            "n_trials": len(study.trials),
            "n_completed_trials": len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            ),
            "n_pruned_trials": len(
                [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
            ),
        }
    )

    # Train final model with best parameters
    print("\nTraining final model with best parameters...")
    best_config = best_trial.params.copy()
    best_config.update({"epochs": 15, "patience": 5})  # More epochs for final model

    # Initialize wandb for final training
    wandb.finish()  # Finish study run
    wandb.init(
        project="ag-news-textcnn-optuna",
        name="final_model",
        config=best_config,
        tags=["textcnn", "ag-news", "optuna", "final"],
    )

    final_result = train_textcnn_single(best_config, train_df_, val_df_)

    if final_result["status"] == "ok":
        # Create output directories
        os.makedirs("outputs/textcnn/models", exist_ok=True)

        # Save best model to standardized location
        torch.save(
            torch.load(final_result["ckpt"], map_location="cpu"),
            "outputs/textcnn/models/best_model.pt",
        )

        # Evaluate on test set
        print("\nEvaluating final model on test set...")
        best_model = TextCNN(
            vocab_size=vocab_size,
            emb_dim=best_config["emb_dim"],
            num_classes=4,
            kernel_sizes=best_config["kernel_sizes"],
            num_channels=best_config["channels"],
            dropout=best_config["dropout"],
            pad_idx=0,
        ).to(device)

        best_model.load_state_dict(
            torch.load("outputs/textcnn/models/best_model.pt", map_location=device)
        )

        # Create test dataset and dataloader
        ds_test = AGNewsDataset(test_df, stoi, max_len=best_config["max_len"])
        dl_test = DataLoader(
            ds_test,
            batch_size=best_config["batch_size"],
            shuffle=False,
            num_workers=num_workers,
            pin_memory=device != "cpu",
        )

        # Evaluate on test set
        crit = nn.CrossEntropyLoss()
        test_loss, test_acc = run_epoch(
            dl_test, best_model, crit, device=device, epoch="Test", verbose=True
        )

        print(f"Final test accuracy: {test_acc:.4f}")
        print(f"Final test loss: {test_loss:.4f}")

        # Log final test results
        wandb.log(
            {
                "final_test_accuracy": test_acc,
                "final_test_loss": test_loss,
                "final_val_accuracy": final_result["val_acc"],
            }
        )

        # Save Optuna study
        study_path = "outputs/textcnn/optuna_study.db"
        os.makedirs(os.path.dirname(study_path), exist_ok=True)
        study.trials_dataframe().to_csv(
            "outputs/textcnn/optuna_trials.csv", index=False
        )
        print(f"Optuna trials saved to: outputs/textcnn/optuna_trials.csv")

    else:
        print("\nFinal training failed with error:")
        print(final_result["msg"])
        wandb.log({"final_training_status": "failed"})

    # Finish final wandb run
    wandb.finish()


if __name__ == "__main__":
    import sys

    n_trials = 20  # default
    if len(sys.argv) > 1:
        try:
            n_trials = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of trials: {sys.argv[1]}. Using default: {n_trials}")

    print(f"Running optimization with {n_trials} trials")
    main(n_trials)
