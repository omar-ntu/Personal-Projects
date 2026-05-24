from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

try:
    import cv2
except ImportError:  # pragma: no cover - handled when RealVideoDataset is used.
    cv2 = None


VIDEO_EXTENSIONS = {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".webm"}
IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | IMAGE_EXTENSIONS


@dataclass
class VideoConfig:
    num_videos: int = 1000
    frames: int = 16
    image_size: int = 32
    input_channels: int = 1
    shape_size: int = 6
    anomaly_prob: float = 0.0
    seed: int = 42


@dataclass
class RealVideoConfig:
    data_root: str
    video_root: str | None = None
    frames: int = 16
    image_size: int = 64
    input_channels: int = 3
    clips_per_video: int = 1
    frame_stride: int = 1
    max_videos: int | None = None
    seed: int = 42


@dataclass
class SomethingSomethingV2Config(RealVideoConfig):
    split: str = "train"
    labels_root: str | None = None
    skip_missing: bool = True


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
    dataset: str = "synthetic",
    data_root: str | None = None,
    video_root: str | None = None,
    split: str = "train",
    num_videos: int = 1000,
    frames: int = 16,
    image_size: int = 32,
    input_channels: int = 1,
    batch_size: int = 32,
    anomaly_prob: float = 0.0,
    clips_per_video: int = 1,
    frame_stride: int = 1,
    max_videos: int | None = None,
    seed: int = 42,
    val_fraction: float = 0.15,
) -> Tuple[DataLoader, DataLoader]:
    dataset_obj = build_dataset(
        dataset=dataset,
        data_root=data_root,
        video_root=video_root,
        split=split,
        num_videos=num_videos,
        frames=frames,
        image_size=image_size,
        input_channels=input_channels,
        anomaly_prob=anomaly_prob,
        clips_per_video=clips_per_video,
        frame_stride=frame_stride,
        max_videos=max_videos,
        seed=seed,
    )
    val_size = max(1, int(len(dataset_obj) * val_fraction))
    train_size = len(dataset_obj) - val_size
    if train_size <= 0:
        raise ValueError("Dataset is too small for a train/validation split.")
    train_set, val_set = random_split(
        dataset_obj,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed),
    )
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def build_dataset(
    dataset: str = "synthetic",
    data_root: str | None = None,
    video_root: str | None = None,
    split: str = "train",
    num_videos: int = 1000,
    frames: int = 16,
    image_size: int = 32,
    input_channels: int = 1,
    anomaly_prob: float = 0.0,
    clips_per_video: int = 1,
    frame_stride: int = 1,
    max_videos: int | None = None,
    seed: int = 42,
) -> Dataset:
    if dataset == "synthetic":
        if input_channels != 1:
            raise ValueError("SyntheticMovingShapesDataset is grayscale; use --input-channels 1.")
        return SyntheticMovingShapesDataset(VideoConfig(
            num_videos=num_videos,
            frames=frames,
            image_size=image_size,
            input_channels=input_channels,
            anomaly_prob=anomaly_prob,
            seed=seed,
        ))
    if dataset == "folder":
        if not data_root:
            raise ValueError("--data-root is required when --dataset folder.")
        return RealVideoFolderDataset(RealVideoConfig(
            data_root=data_root,
            video_root=video_root,
            frames=frames,
            image_size=image_size,
            input_channels=input_channels,
            clips_per_video=clips_per_video,
            frame_stride=frame_stride,
            max_videos=max_videos,
            seed=seed,
        ))
    if dataset == "something-v2":
        if not data_root:
            raise ValueError("--data-root is required when --dataset something-v2.")
        return SomethingSomethingV2Dataset(SomethingSomethingV2Config(
            data_root=data_root,
            video_root=video_root,
            split=split,
            frames=frames,
            image_size=image_size,
            input_channels=input_channels,
            clips_per_video=clips_per_video,
            frame_stride=frame_stride,
            max_videos=max_videos,
            seed=seed,
        ))
    raise ValueError(f"Unknown dataset: {dataset}")


class RealVideoFolderDataset(Dataset):
    """Read real videos or frame folders from disk.

    Supported layouts:
        data_root/class_name/sample.mp4
        data_root/class_name/sample_frames/000001.jpg
        data_root/sample.mp4

    The returned video tensor has shape [C, T, H, W] with values in [0, 1].
    """

    def __init__(self, config: RealVideoConfig):
        if cv2 is None:
            raise ImportError("RealVideoFolderDataset requires opencv-python. Run `pip install -r requirements.txt`.")
        self.config = config
        self.root = Path(config.data_root)
        if not self.root.exists():
            raise FileNotFoundError(f"Video data root does not exist: {self.root}")
        self.class_to_idx = self._discover_classes()
        self.samples = self._discover_samples()
        if not self.samples:
            raise FileNotFoundError(f"No video files or frame folders found under: {self.root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        sample = self.samples[idx]
        path = sample["path"]
        if path.is_dir():
            frames = self._read_frame_folder(path)
            clip = self._sample_clip(frames, int(sample["clip_index"]))
        else:
            clip = self._read_video_clip(path, int(sample["clip_index"]))
        clip = self._resize_and_format(clip)
        return {
            "video": torch.from_numpy(clip).float(),
            "label": torch.tensor(int(sample["label"]), dtype=torch.long),
            "is_anomaly": torch.tensor(0, dtype=torch.long),
            "path": str(path),
        }

    def _discover_classes(self) -> Dict[str, int]:
        class_names = sorted({
            p.relative_to(self.root).parts[0]
            for p in self.root.rglob("*")
            if p.is_file() and p.suffix.lower() in MEDIA_EXTENSIONS and len(p.relative_to(self.root).parts) > 1
        })
        return {name: idx for idx, name in enumerate(class_names)}

    def _discover_samples(self) -> List[Dict[str, Path | int]]:
        video_paths = sorted(p for p in self.root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
        frame_dirs = self._discover_frame_dirs()
        paths: Sequence[Path] = [*video_paths, *frame_dirs]
        if self.config.max_videos is not None:
            paths = paths[: self.config.max_videos]

        samples: List[Dict[str, Path | int]] = []
        for path in paths:
            label = self._label_for_path(path)
            for clip_index in range(max(1, self.config.clips_per_video)):
                samples.append({"path": path, "label": label, "clip_index": clip_index})
        return samples

    def _discover_frame_dirs(self) -> List[Path]:
        dirs = []
        for directory in sorted(p for p in self.root.rglob("*") if p.is_dir()):
            if any(child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS for child in directory.iterdir()):
                dirs.append(directory)
        return dirs

    def _label_for_path(self, path: Path) -> int:
        rel = path.relative_to(self.root)
        if len(rel.parts) <= 1:
            return 0
        return self.class_to_idx.get(rel.parts[0], 0)

    def _read_video_clip(self, path: Path, clip_index: int) -> np.ndarray:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video file: {path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            total = self._count_video_frames(cap)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        indices = self._clip_indices(total, clip_index)
        frames = []
        try:
            last_index = -1
            for frame_index in indices:
                if frame_index <= last_index:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                else:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = cap.read()
                if not ok:
                    frame = frames[-1] if frames else np.zeros((self.config.image_size, self.config.image_size, 3), dtype=np.uint8)
                    frames.append(frame)
                    continue
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                last_index = int(frame_index)
        finally:
            cap.release()
        if not frames:
            raise RuntimeError(f"Video file contained no decodable frames: {path}")
        return np.stack(frames, axis=0)

    @staticmethod
    def _count_video_frames(cap) -> int:
        total = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            total += 1
        return total

    def _read_frame_folder(self, path: Path) -> np.ndarray:
        image_paths = sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
        frames = []
        for image_path in image_paths:
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            frames.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not frames:
            raise RuntimeError(f"Frame folder contained no decodable images: {path}")
        return np.stack(frames, axis=0)

    def _sample_clip(self, frames: np.ndarray, clip_index: int) -> np.ndarray:
        indices = self._clip_indices(frames.shape[0], clip_index)
        return frames[indices]

    def _clip_indices(self, total: int, clip_index: int) -> np.ndarray:
        if total <= 0:
            raise RuntimeError("Cannot sample a clip from an empty video.")
        needed = self.config.frames * self.config.frame_stride
        if total >= needed:
            max_start = total - needed
            if self.config.clips_per_video <= 1:
                start = max_start // 2
            else:
                start = round(max_start * clip_index / max(1, self.config.clips_per_video - 1))
            indices = start + np.arange(self.config.frames) * self.config.frame_stride
        else:
            indices = np.linspace(0, total - 1, self.config.frames).round().astype(np.int64)
        return indices.astype(np.int64)

    def _resize_and_format(self, clip: np.ndarray) -> np.ndarray:
        resized = [
            cv2.resize(frame, (self.config.image_size, self.config.image_size), interpolation=cv2.INTER_AREA)
            for frame in clip
        ]
        arr = np.stack(resized, axis=0).astype(np.float32) / 255.0
        if self.config.input_channels == 1:
            arr = np.stack([cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) for frame in arr], axis=0)
            arr = arr[:, None, :, :].astype(np.float32) / 255.0
        elif self.config.input_channels == 3:
            arr = arr.transpose(0, 3, 1, 2)
        else:
            raise ValueError("--input-channels must be 1 or 3.")
        return arr.transpose(1, 0, 2, 3)


class SomethingSomethingV2Dataset(RealVideoFolderDataset):
    """Something-Something V2 reader using official label JSON files.

    Expected layout after preparation:
        data/something_something_v2/
          labels/train.json
          labels/validation.json
          labels/labels.json
          20bn-something-something-v2/78687.webm
    """

    def __init__(self, config: SomethingSomethingV2Config):
        if cv2 is None:
            raise ImportError("SomethingSomethingV2Dataset requires opencv-python. Run `pip install -r requirements.txt`.")
        self.config = config
        self.root = Path(config.data_root)
        labels_root = Path(config.labels_root) if config.labels_root else self.root / "labels"
        split_path = labels_root / f"{config.split}.json"
        label_map_path = labels_root / "labels.json"
        if not split_path.exists():
            raise FileNotFoundError(f"Something-Something split file not found: {split_path}")
        if not label_map_path.exists():
            raise FileNotFoundError(f"Something-Something label map not found: {label_map_path}")

        self.video_root = self._resolve_video_root(config.video_root)
        self.class_to_idx = self._load_label_map(label_map_path)
        self.samples = self._discover_labeled_samples(split_path)
        if not self.samples:
            raise FileNotFoundError(
                "No matching Something-Something V2 videos were found. "
                f"Looked under {self.video_root}; extract videos or lower --max-videos to match an extracted subset."
            )

    def _resolve_video_root(self, video_root: str | None) -> Path:
        if video_root:
            return Path(video_root)
        preferred = self.root / "20bn-something-something-v2"
        if preferred.exists():
            return preferred
        videos = self.root / "videos"
        if videos.exists():
            return videos
        return self.root

    @staticmethod
    def _load_label_map(path: Path) -> Dict[str, int]:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {str(k): int(v) for k, v in raw.items()}

    def _discover_labeled_samples(self, split_path: Path) -> List[Dict[str, Path | int]]:
        with open(split_path, "r", encoding="utf-8") as f:
            rows = json.load(f)
        if self.config.max_videos is not None:
            rows = rows[: self.config.max_videos]

        video_by_stem = {
            path.stem: path
            for path in self.video_root.rglob("*")
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        }
        samples: List[Dict[str, Path | int]] = []
        for row in rows:
            video_id = str(row["id"])
            path = video_by_stem.get(video_id)
            if path is None:
                if self.config.skip_missing:
                    continue
                path = self.video_root / f"{video_id}.webm"
            label = self._label_for_template(str(row["template"]))
            for clip_index in range(max(1, self.config.clips_per_video)):
                samples.append({"path": path, "label": label, "clip_index": clip_index})
        return samples

    def _label_for_template(self, template: str) -> int:
        normalized = template.replace("[", "").replace("]", "")
        if normalized in self.class_to_idx:
            return self.class_to_idx[normalized]
        return self.class_to_idx.get(normalized[:1].upper() + normalized[1:], 0)
