from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


@dataclass
class VideoConfig:
    num_videos: int = 1000
    frames: int = 16
    image_size: int = 32
    shape_size: int = 6
    anomaly_prob: float = 0.0
    seed: int = 42


class SyntheticMovingShapesDataset(Dataset):
    """Small synthetic video dataset for video-reconstruction experiments.

    Each sample is a short grayscale clip containing a simple moving object. The
    clips are generated deterministically from `seed + index`, so no dataset files
    are required. This makes the project runnable on a laptop while preserving the
    main research idea: compressing temporal visual information into a latent code.

    Output dict:
        video: FloatTensor with shape [1, T, H, W], values in [0, 1]
        label: int in {0, 1, 2}; square, circle, diamond
        is_anomaly: int; whether a sudden jump/occlusion was injected
    """

    SHAPES = ("square", "circle", "diamond")

    def __init__(self, config: VideoConfig):
        self.config = config

    def __len__(self) -> int:
        return self.config.num_videos

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        rng = np.random.default_rng(self.config.seed + idx)
        video, label, is_anomaly = self._generate_clip(rng)
        return {
            "video": torch.from_numpy(video).float(),
            "label": torch.tensor(label, dtype=torch.long),
            "is_anomaly": torch.tensor(is_anomaly, dtype=torch.long),
        }

    def _generate_clip(self, rng: np.random.Generator) -> Tuple[np.ndarray, int, int]:
        T = self.config.frames
        H = W = self.config.image_size
        s = self.config.shape_size
        margin = s + 1

        label = int(rng.integers(0, len(self.SHAPES)))
        shape_name = self.SHAPES[label]

        # Start position and velocity in pixel coordinates.
        x = float(rng.integers(margin, W - margin))
        y = float(rng.integers(margin, H - margin))
        vx = float(rng.choice([-2, -1, 1, 2]))
        vy = float(rng.choice([-2, -1, 1, 2]))

        is_anomaly = int(rng.random() < self.config.anomaly_prob)
        jump_t = int(rng.integers(T // 3, max(T // 3 + 1, 2 * T // 3))) if is_anomaly else -1
        occlude_t = int(rng.integers(0, T)) if is_anomaly and rng.random() < 0.5 else -1

        frames = []
        for t in range(T):
            if t == jump_t:
                x = float(rng.integers(margin, W - margin))
                y = float(rng.integers(margin, H - margin))

            canvas = np.zeros((H, W), dtype=np.float32)
            if t != occlude_t:
                self._draw_shape(canvas, shape_name, int(round(x)), int(round(y)), s)

            # Add very mild sensor-like noise.
            noise = rng.normal(0.0, 0.015, size=(H, W)).astype(np.float32)
            canvas = np.clip(canvas + noise, 0.0, 1.0)
            frames.append(canvas)

            x += vx
            y += vy
            if x < margin or x > W - margin:
                vx *= -1
                x = np.clip(x, margin, W - margin)
            if y < margin or y > H - margin:
                vy *= -1
                y = np.clip(y, margin, H - margin)

        arr = np.stack(frames, axis=0)[None, ...]  # [1, T, H, W]
        return arr, label, is_anomaly

    @staticmethod
    def _draw_shape(canvas: np.ndarray, shape_name: str, cx: int, cy: int, size: int) -> None:
        H, W = canvas.shape
        yy, xx = np.mgrid[0:H, 0:W]
        if shape_name == "square":
            mask = (np.abs(xx - cx) <= size // 2) & (np.abs(yy - cy) <= size // 2)
        elif shape_name == "circle":
            mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= (size // 2 + 1) ** 2
        else:  # diamond
            mask = np.abs(xx - cx) + np.abs(yy - cy) <= size
        canvas[mask] = 1.0


def build_dataloaders(
    num_videos: int = 1000,
    frames: int = 16,
    image_size: int = 32,
    batch_size: int = 32,
    anomaly_prob: float = 0.0,
    seed: int = 42,
    val_fraction: float = 0.15,
) -> Tuple[DataLoader, DataLoader]:
    config = VideoConfig(
        num_videos=num_videos,
        frames=frames,
        image_size=image_size,
        anomaly_prob=anomaly_prob,
        seed=seed,
    )
    dataset = SyntheticMovingShapesDataset(config)
    val_size = max(1, int(num_videos * val_fraction))
    train_size = num_videos - val_size
    train_set, val_set = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader
