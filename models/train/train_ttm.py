"""
TTM (Tiny Time Mixer) for Phase 1 benchmark.

Reads:  D:/updated_dataset/models/windows/sequence_windows_all.npz
Writes: D:/updated_dataset/models/weights/ttm/ttm_config.json
        (appends to model_scores_all.csv, model_metrics_all.csv)

TTM requires: tsfm_public (IBM time-series foundation models)
    pip install tsfm-public

If tsfm_public is unavailable or any runtime error occurs, the model is
marked NOT_RUN with the exact reason. No fake scores are ever written.
The x_hat = x fallback is explicitly forbidden — if forward fails the
run is aborted.
"""
import sys
import json
import time
from pathlib import Path

import numpy as np

ROOT = Path(r"D:\updated_dataset")
sys.path.insert(0, str(ROOT / "models"))
sys.path.insert(0, str(ROOT / "models" / "scripts_common"))

from common_paths import (
    SEQ_ALL_NPZ, RESULTS_DIR, WEIGHTS_DIR, ensure_all_dirs,
)
from model_utils import choose_threshold, compute_metrics, save_scores, save_metrics_row

MODEL_NAME = "ttm"


def _try_import_ttm():
    """Attempt to import TTM from tsfm_public or transformers."""
    try:
        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction
        return TinyTimeMixerForPrediction, "tsfm_public"
    except ImportError:
        pass
    try:
        from transformers import TinyTimeMixerForPrediction
        return TinyTimeMixerForPrediction, "transformers"
    except ImportError:
        pass
    return None, None


def _not_run(reason: str, weights_dir: Path, elapsed: float, feature_count: int):
    config = {
        "model_name": MODEL_NAME,
        "status": "NOT_RUN",
        "reason": reason,
        "install_hint": "pip install tsfm-public",
        "elapsed_s": round(elapsed, 2),
    }
    with open(weights_dir / "ttm_config.json", "w") as f:
        json.dump(config, f, indent=2)
    save_metrics_row(
        {"precision": float("nan"), "recall": float("nan"), "f1": float("nan"),
         "roc_auc": float("nan"), "pr_auc": float("nan"),
         "accuracy": float("nan"), "balanced_accuracy": float("nan"),
         "tp": 0, "fp": 0, "tn": 0, "fn": 0,
         "false_positive_count": 0, "false_negative_count": 0},
        MODEL_NAME, "primary", "NOT_RUN",
        {"failure_reason": reason, "feature_count": feature_count,
         "runtime_seconds": round(elapsed, 2), "artifact_path": ""},
        RESULTS_DIR,
    )
    print(f"  TTM: NOT_RUN — {reason}")
    return config


def _extract_reconstruction(out, model_input):
    """Extract reconstruction tensor from TTM output.

    Raises RuntimeError if the output type is not understood.
    Never falls back to returning the input unchanged.
    """
    import torch
    if isinstance(out, torch.Tensor):
        if out.shape != model_input.shape:
            raise RuntimeError(
                f"TTM forward returned tensor with shape {out.shape}, "
                f"but input shape is {model_input.shape}. "
                f"Cannot use as reconstruction without shape match."
            )
        return out
    if hasattr(out, "reconstruction"):
        recon = out.reconstruction
        if recon is None:
            raise RuntimeError("TTM output.reconstruction is None.")
        return recon
    if hasattr(out, "last_hidden_state"):
        raise RuntimeError(
            "TTM returned last_hidden_state instead of a reconstruction. "
            "This model does not support direct reconstruction output. "
            "Use a model variant with reconstruction head."
        )
    raise RuntimeError(
        f"TTM forward returned unexpected type {type(out).__name__}. "
        f"Cannot extract reconstruction. "
        f"Attributes: {[a for a in dir(out) if not a.startswith('_')][:20]}"
    )


def train_ttm():
    ensure_all_dirs()
    t0 = time.time()
    weights_dir = WEIGHTS_DIR / "ttm"

    npz = SEQ_ALL_NPZ
    if not npz.exists():
        return _not_run("sequence_windows_all.npz not found", weights_dir, time.time() - t0, 0)

    TTMClass, source = _try_import_ttm()
    if TTMClass is None:
        return _not_run(
            "tsfm_public not installed — run: pip install tsfm-public",
            weights_dir, time.time() - t0, 22)

    # Load data
    data = np.load(str(npz), allow_pickle=True)

    if "source_dataset" not in data:
        return _not_run(
            "source_dataset array missing from NPZ. "
            "Re-run 00_build_model_windows.py.",
            weights_dir, time.time() - t0, 0)

    seqs = data["sequences"].astype(np.float32)
    y = data["y_anomaly"].astype(int)
    splits = data["splits"]
    source_dataset = data["source_dataset"]

    # Training: clean-source normal windows only
    train_mask = (splits == "train") & (y == 0) & (source_dataset == "clean")
    if int(train_mask.sum()) == 0:
        return _not_run(
            "No clean-source normal training windows found in NPZ.",
            weights_dir, time.time() - t0, 0)

    val_mask = splits == "val"
    test_mask = splits == "test"

    X_train = seqs[train_mask]
    X_val = seqs[val_mask]
    y_val = y[val_mask]
    X_test = seqs[test_mask]
    y_test = y[test_mask]
    wids_test = data["window_ids"][test_mask]

    n_time = seqs.shape[1]
    n_feat = seqs.shape[2]

    # Normalize using clean-source training statistics
    flat = X_train.reshape(-1, n_feat)
    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    print(f"  TTM [{source}]: n_time={n_time}, n_feat={n_feat}, "
          f"train={len(X_train)}, val={len(X_val)}")

    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        config_kwargs = {
            "context_length": n_time,
            "patch_length": 8,
            "num_input_channels": n_feat,
            "d_model": 64,
            "num_layers": 3,
            "dropout": 0.1,
        }
        try:
            model = TTMClass(**config_kwargs)
        except TypeError as e:
            # Minimal config fallback — still fails hard if this also fails
            try:
                model = TTMClass(context_length=n_time, prediction_length=n_time)
            except Exception as e2:
                return _not_run(
                    f"TTM model instantiation failed. "
                    f"Full config error: {e}. "
                    f"Minimal config error: {e2}.",
                    weights_dir, time.time() - t0, n_feat)

        X_tr_t = torch.from_numpy(X_train)
        X_va_t = torch.from_numpy(X_val)
        X_te_t = torch.from_numpy(X_test)
        y_va_t = torch.from_numpy(y_val.astype(np.float32))
        y_te_t = torch.from_numpy(y_test.astype(np.float32))

        tr_ds = TensorDataset(X_tr_t)
        va_ds = TensorDataset(X_va_t, y_va_t)
        te_ds = TensorDataset(X_te_t, y_te_t)
        tr_loader = DataLoader(tr_ds, batch_size=128, shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=128, shuffle=False)
        te_loader = DataLoader(te_ds, batch_size=128, shuffle=False)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.MSELoss()
        best_val = float("inf")
        best_state = None
        patience_count = 0
        PATIENCE = 10

        for epoch in range(100):
            model.train()
            for (x,) in tr_loader:
                optimizer.zero_grad()
                # _extract_reconstruction raises on bad output — no fallback
                out = model(x)
                x_hat = _extract_reconstruction(out, x)
                loss = loss_fn(x_hat, x)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            model.eval()
            va_losses = []
            with torch.no_grad():
                for (x, _) in va_loader:
                    out = model(x)
                    x_hat = _extract_reconstruction(out, x)
                    va_losses.append(loss_fn(x_hat, x).item())
            va_l = float(np.mean(va_losses))
            if va_l < best_val - 1e-6:
                best_val = va_l
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
            if patience_count >= PATIENCE:
                print(f"    TTM early stop epoch {epoch+1}")
                break

        if best_state:
            model.load_state_dict(best_state)

        # Score — no fallback; failures propagate to outer except
        model.eval()
        scores_va_list, scores_te_list = [], []
        with torch.no_grad():
            for (x, _) in va_loader:
                out = model(x)
                x_hat = _extract_reconstruction(out, x)
                err = torch.nn.functional.mse_loss(
                    x_hat, x, reduction="none").mean(dim=(1, 2))
                scores_va_list.append(err.numpy())
            for (x, _) in te_loader:
                out = model(x)
                x_hat = _extract_reconstruction(out, x)
                err = torch.nn.functional.mse_loss(
                    x_hat, x, reduction="none").mean(dim=(1, 2))
                scores_te_list.append(err.numpy())

        scores_va = np.concatenate(scores_va_list)
        scores_te = np.concatenate(scores_te_list)
        best_thr, _ = choose_threshold(scores_va, y_val)
        y_pred_te = (scores_te >= best_thr).astype(int)
        metrics = compute_metrics(y_test, y_pred_te, scores_te)
        elapsed = time.time() - t0

        model_path = weights_dir / "ttm_best.pt"
        torch.save({"model_state": model.state_dict(), "mean": mean, "std": std,
                    "threshold": best_thr, "source": source}, model_path)

        config = {
            "model_name": MODEL_NAME, "status": "PASS", "source": source,
            "threshold": best_thr, "test_metrics": metrics,
            "elapsed_s": round(elapsed, 2), "artifact_path": str(model_path),
        }
        with open(weights_dir / "ttm_config.json", "w") as f:
            json.dump(config, f, indent=2)

        save_scores(wids_test, y_test, y_pred_te, scores_te, MODEL_NAME, "primary", RESULTS_DIR)
        save_metrics_row(metrics, MODEL_NAME, "primary", "PASS",
                         {"feature_count": n_feat, "source": source,
                          "threshold": best_thr, "runtime_seconds": round(elapsed, 2),
                          "artifact_path": str(model_path)},
                         RESULTS_DIR)

        print(f"  TTM: F1={metrics['f1']:.4f} P={metrics['precision']:.4f} "
              f"R={metrics['recall']:.4f} ROC-AUC={metrics['roc_auc']:.4f}")
        return config

    except Exception as e:
        # Exact exception — no fake scores written
        reason = f"{type(e).__name__}: {e}"
        return _not_run(f"Runtime error — {reason}", weights_dir, time.time() - t0, n_feat)


def main():
    print("Training TTM...")
    train_ttm()
    print("TTM complete.")


if __name__ == "__main__":
    main()
