from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import matplotlib.pyplot as plt
import torch

from .utils import ensure_dir


def _to_numpy_video(video: torch.Tensor):
    """Convert [1,T,H,W] or [T,H,W] tensor to numpy [T,H,W]."""
    video = video.detach().cpu().float().clamp(0, 1)
    if video.ndim == 4:
        video = video[0]
    return video.numpy()


def save_video_strip(video: torch.Tensor, path: str | Path, title: str = "video", max_frames: int = 16) -> None:
    arr = _to_numpy_video(video)
    t = min(arr.shape[0], max_frames)
    ensure_dir(Path(path).parent)
    fig, axes = plt.subplots(1, t, figsize=(1.4 * t, 1.6))
    if t == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        ax.imshow(arr[i], cmap="gray", vmin=0, vmax=1)
        ax.set_title(str(i), fontsize=8)
        ax.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_reconstruction_comparison(input_video: torch.Tensor, recon_video: torch.Tensor, path: str | Path, max_frames: int = 16) -> None:
    x = _to_numpy_video(input_video)
    r = _to_numpy_video(recon_video)
    e = abs(x - r)
    t = min(x.shape[0], max_frames)
    ensure_dir(Path(path).parent)
    fig, axes = plt.subplots(3, t, figsize=(1.4 * t, 4.6))
    rows = [(x, "input"), (r, "reconstruction"), (e, "absolute error")]
    for row_idx, (arr, label) in enumerate(rows):
        for i in range(t):
            ax = axes[row_idx, i]
            ax.imshow(arr[i], cmap="gray", vmin=0, vmax=1)
            if i == 0:
                ax.set_ylabel(label, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_interpolation_grid(videos: Iterable[torch.Tensor], path: str | Path, labels: List[str] | None = None, frame_index: int = 8) -> None:
    vids = [_to_numpy_video(v) for v in videos]
    ensure_dir(Path(path).parent)
    n = len(vids)
    fig, axes = plt.subplots(1, n, figsize=(1.8 * n, 2.0))
    if n == 1:
        axes = [axes]
    for i, (ax, arr) in enumerate(zip(axes, vids)):
        f = min(frame_index, arr.shape[0] - 1)
        ax.imshow(arr[f], cmap="gray", vmin=0, vmax=1)
        ax.set_title(labels[i] if labels else f"{i}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Latent interpolation: one middle frame from each decoded clip")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_keyframe_strip(video: torch.Tensor, indices: List[int], path: str | Path) -> None:
    arr = _to_numpy_video(video)
    ensure_dir(Path(path).parent)
    fig, axes = plt.subplots(1, len(indices), figsize=(1.8 * len(indices), 1.9))
    if len(indices) == 1:
        axes = [axes]
    for ax, idx in zip(axes, indices):
        ax.imshow(arr[idx], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"t={idx}", fontsize=8)
        ax.axis("off")
    fig.suptitle("Latent keyframes")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
