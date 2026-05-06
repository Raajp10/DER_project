"""PyTorch training utilities for autoencoder models.

Shared DataLoader, training loop, early stopping, and loss tracking.
All models train on normal windows only using MSE reconstruction loss.
"""
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def make_seq_dataloaders(seq_npz_path, batch_size: int = 128,
                         train_split: str = "train",
                         val_split: str = "val") -> tuple:
    """Load sequence tensors from npz and create DataLoaders.

    Training filter: source_dataset == 'clean' AND y_anomaly == 0.
    Raises RuntimeError if source_dataset array is missing from the NPZ.

    Returns (train_loader, val_loader, test_loader, n_time, n_features,
             mean, std, wids_test)
    """
    data = np.load(str(seq_npz_path), allow_pickle=True)
    seqs = data["sequences"].astype(np.float32)
    y = data["y_anomaly"].astype(int)
    splits = data["splits"]

    if "source_dataset" not in data:
        raise RuntimeError(
            "source_dataset array is missing from the NPZ file. "
            "Re-run 00_build_model_windows.py to regenerate sequence windows."
        )
    source_dataset = data["source_dataset"]

    # Train: clean-source normal windows only (no contamination from attacked-source)
    train_mask = (splits == train_split) & (y == 0) & (source_dataset == "clean")
    n_clean_train = int(train_mask.sum())
    if n_clean_train == 0:
        raise RuntimeError(
            f"No clean-source normal training windows found in NPZ. "
            f"splits=='{train_split}': {(splits == train_split).sum()}, "
            f"y==0: {(y == 0).sum()}, source_dataset=='clean': {(source_dataset == 'clean').sum()}."
        )

    val_mask = splits == val_split
    test_mask = splits == "test"

    X_train = torch.from_numpy(seqs[train_mask])
    X_val = torch.from_numpy(seqs[val_mask])
    y_val = torch.from_numpy(y[val_mask].astype(np.float32))
    X_test = torch.from_numpy(seqs[test_mask])
    y_test = torch.from_numpy(y[test_mask].astype(np.float32))

    # Compute per-feature mean/std on training normal data for normalization
    flat = seqs[train_mask].reshape(-1, seqs.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-8

    mean_t = torch.from_numpy(mean)
    std_t = torch.from_numpy(std)

    X_train = (X_train - mean_t) / std_t
    X_val = (X_val - mean_t) / std_t
    X_test = (X_test - mean_t) / std_t

    train_ds = TensorDataset(X_train)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    n_time = seqs.shape[1]
    n_feat = seqs.shape[2]
    wids_test = data["window_ids"][test_mask]

    return train_loader, val_loader, test_loader, n_time, n_feat, mean, std, wids_test


class EarlyStopper:
    """Early stopping with patience."""
    def __init__(self, patience: int = 10, min_delta: float = 1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val = float("inf")
        self.counter = 0

    def step(self, val_loss: float) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_val - self.min_delta:
            self.best_val = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def train_autoencoder(model: nn.Module, train_loader: DataLoader,
                      val_loader: DataLoader, max_epochs: int = 100,
                      lr: float = 1e-3, patience: int = 10,
                      loss_fn=None) -> tuple:
    """Train an autoencoder and return (train_losses, val_losses, best_epoch).

    Model input/output: (batch, time, features) for sequence models
    or (batch, features) for flat models.
    """
    if loss_fn is None:
        loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    stopper = EarlyStopper(patience=patience)
    train_losses, val_losses = [], []
    best_state = None
    best_epoch = 0

    for epoch in range(max_epochs):
        model.train()
        batch_losses = []
        for batch in train_loader:
            x = batch[0]
            optimizer.zero_grad()
            x_hat = model(x)
            loss = loss_fn(x_hat, x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            batch_losses.append(loss.item())
        train_loss = float(np.mean(batch_losses))
        train_losses.append(train_loss)

        model.eval()
        val_batch = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch[0]
                x_hat = model(x)
                val_batch.append(loss_fn(x_hat, x).item())
        val_loss = float(np.mean(val_batch))
        val_losses.append(val_loss)

        if val_loss <= stopper.best_val:
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: train={train_loss:.6f}, val={val_loss:.6f}")

        if stopper.step(val_loss):
            print(f"    Early stop at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return train_losses, val_losses, best_epoch


def score_autoencoder(model: nn.Module, loader: DataLoader,
                      return_y: bool = True) -> tuple:
    """Get per-sample reconstruction scores from autoencoder.

    Returns (scores, y_true) where scores = per-sample MSE.
    """
    model.eval()
    all_scores, all_y = [], []
    loss_fn = nn.MSELoss(reduction="none")
    with torch.no_grad():
        for batch in loader:
            x = batch[0]
            y = batch[1] if len(batch) > 1 else None
            x_hat = model(x)
            # Per-sample MSE averaged over all dimensions
            err = loss_fn(x_hat, x).mean(dim=tuple(range(1, x.ndim)))
            all_scores.append(err.numpy())
            if y is not None:
                all_y.append(y.numpy())
    scores = np.concatenate(all_scores)
    y_out = np.concatenate(all_y) if all_y else None
    return scores, y_out
