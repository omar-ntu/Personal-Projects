from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch

from .data import build_dataset
from .losses import reconstruction_metrics
from .models import FrameVAE, TemporalVAE
from .utils import get_device, load_checkpoint, save_json


def build_model_from_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model_type = cfg.get("model_type")
    if model_type == "frame_vae":
        model = FrameVAE(
            latent_dim=int(cfg["latent_dim"]),
            image_size=int(cfg.get("image_size", 32)),
            input_channels=int(cfg.get("input_channels", 1)),
        ).to(device)
    elif model_type == "temporal_vae":
        model = TemporalVAE(
            latent_dim=int(cfg["latent_dim"]),
            frames=int(cfg.get("frames", 16)),
            image_size=int(cfg.get("image_size", 32)),
            input_channels=int(cfg.get("input_channels", 1)),
        ).to(device)
    else:
        raise ValueError(f"Unknown model_type in checkpoint: {model_type}")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


@torch.no_grad()
def reconstruct_deterministic(model, video: torch.Tensor) -> torch.Tensor:
    """Reconstruct with the VAE mean latent instead of sampling random noise."""
    z = model.encode_video_mu(video)
    return model.decode_video_latents(z)


@torch.no_grad()
def evaluate_checkpoint(args: argparse.Namespace, ckpt_path: str, device: torch.device) -> Dict[str, float]:
    model, cfg = build_model_from_checkpoint(ckpt_path, device)
    dataset = build_dataset(
        dataset=args.dataset,
        data_root=args.data_root,
        video_root=args.video_root,
        split=args.split,
        num_videos=args.num_videos,
        frames=int(cfg.get("frames", 16)),
        image_size=int(cfg.get("image_size", 32)),
        input_channels=int(cfg.get("input_channels", 1)),
        anomaly_prob=args.anomaly_prob,
        clips_per_video=args.clips_per_video,
        frame_stride=args.frame_stride,
        max_videos=args.max_videos,
        seed=args.seed,
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    totals = {"mse": 0.0, "psnr": 0.0, "ssim_global": 0.0, "temporal_mse": 0.0}
    for batch in loader:
        video = batch["video"].to(device)
        recon = reconstruct_deterministic(model, video)
        metrics = reconstruction_metrics(recon, video)
        for k in totals:
            totals[k] += metrics[k]
    return {k: v / max(1, len(loader)) for k, v in totals.items()}


def main(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    results = {}
    for name, path in [("frame_vae", args.frame_ckpt), ("temporal_vae", args.temporal_ckpt)]:
        if path and Path(path).exists():
            results[name] = evaluate_checkpoint(
                args=args,
                ckpt_path=path,
                device=device,
            )
        else:
            results[name] = {"status": f"checkpoint not found: {path}"}
    save_json(results, args.out)
    print(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained video VAEs on synthetic or folder-based videos.")
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "folder", "something-v2"])
    parser.add_argument("--data-root", type=str, default=None, help="Root containing class folders, videos, frame folders, or Something-Something labels.")
    parser.add_argument("--video-root", type=str, default=None, help="Optional video directory override for Something-Something V2.")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split for Something-Something V2.")
    parser.add_argument("--frame-ckpt", type=str, default="checkpoints/frame_vae.pt")
    parser.add_argument("--temporal-ckpt", type=str, default="checkpoints/temporal_vae.pt")
    parser.add_argument("--num-videos", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--anomaly-prob", type=float, default=0.0, help="Use >0 to evaluate on anomaly/noisy clips.")
    parser.add_argument("--clips-per-video", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out", type=str, default="outputs/evaluation.json")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
