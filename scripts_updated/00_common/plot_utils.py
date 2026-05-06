"""Shared plotting utilities for the DER dataset pipeline."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path


# Academic style defaults
STYLE = {
    "figure.figsize": (14, 8),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "lines.linewidth": 1.2,
}


def apply_style():
    for k, v in STYLE.items():
        plt.rcParams[k] = v


def save_fig(fig: plt.Figure, path: Path, dpi: int = 300):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    size_kb = path.stat().st_size // 1024
    print(f"  Saved {path.name} ({size_kb} KB)")


def time_axis_hours(n: int, total_s: int = 604_800) -> np.ndarray:
    """Return x-axis in hours for n evenly spaced points."""
    return np.linspace(0, total_s / 3600, n)


def downsample(arr: np.ndarray, factor: int = 60) -> np.ndarray:
    """Downsample array by averaging factor-length blocks (for plotting)."""
    n = (len(arr) // factor) * factor
    return arr[:n].reshape(-1, factor).mean(axis=1)


def multi_subplot_time(
    series_list: list,
    labels: list,
    ylabel_list: list,
    title: str,
    path: Path,
    colors: list = None,
    total_s: int = 604_800,
    ds_factor: int = 60,
):
    """Plot multiple time series in stacked subplots, saved to path."""
    apply_style()
    n = len(series_list)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    palette = colors or plt.cm.tab10.colors
    x = time_axis_hours(len(downsample(series_list[0], ds_factor)), total_s)
    for i, (arr, lbl, ylab) in enumerate(zip(series_list, labels, ylabel_list)):
        y = downsample(np.asarray(arr, dtype=float), ds_factor)
        axes[i].plot(x, y, color=palette[i % len(palette)], label=lbl, linewidth=1.0)
        axes[i].set_ylabel(ylab, fontsize=10)
        axes[i].legend(loc="upper right")
    axes[-1].set_xlabel("Time (hours)")
    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()
    save_fig(fig, path)


def bar_chart(labels: list, values: list, title: str, ylabel: str, path: Path, color: str = "#2196F3"):
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(labels))
    ax.bar(x, values, color=color, alpha=0.85, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.tight_layout()
    save_fig(fig, path)


def pass_fail_bar(categories: list, passed: list, failed: list, title: str, path: Path):
    apply_style()
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(categories))
    w = 0.35
    ax.bar(x - w / 2, passed, w, label="PASS", color="#4CAF50", alpha=0.85)
    ax.bar(x + w / 2, failed, w, label="FAIL", color="#F44336", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=20, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    save_fig(fig, path)
