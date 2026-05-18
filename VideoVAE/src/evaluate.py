from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch

from .data import VideoConfig, SyntheticMovingShapesDataset
from .losses import reconstruction_metrics
from .models import FrameVAE, TemporalVAE
from .utils import get_device, load_checkpoint, save_json


def build_model_from_checkpoint(ckpt_path: str, device: torch.device):
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    cfg = ckpt["config"]
    model_type = cfg.get("model_type")
    if model_type == "frame_vae":
        model = FrameVAE(latent_dim=int(cfg["latent_dim"]), image_size=int(cfg.get("image_size", 32))).to(device)
    elif model_type == "temporal_vae":
        model = TemporalVAE(latent_dim=int(cfg["latent_dim"]), frames=int(cfg.get("frames", 16)), image_size=int(cfg.get("image_size", 32))).to(device)
    else:
        raise ValueError(f"Unknown model_type in checkpoint: {model_type}")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, cfg


@torch.no_grad()
def evaluate_checkpoint(ckpt_path: str, num_videos: int, batch_size: int, anomaly_prob: float, seed: int, device: torch.device) -> Dict[str, float]:
    model, cfg = build_model_from_checkpoint(ckpt_path, device)
    dataset = SyntheticMovingShapesDataset(VideoConfig(
        num_videos=num_videos,
        frames=int(cfg.get("frames", 16)),
        image_size=int(cfg.get("image_size", 32)),
        anomaly_prob=anomaly_prob,
        seed=seed,
    ))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    totals = {"mse": 0.0, "psnr": 0.0, "ssim_global": 0.0, "temporal_mse": 0.0}
    for batch in loader:
        video = batch["video"].to(device)
        recon, _, _ = model(video)
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
                path,
                num_videos=args.num_videos,
                batch_size=args.batch_size,
                anomaly_prob=args.anomaly_prob,
                seed=args.seed,
                device=device,
            )
        else:
            results[name] = {"status": f"checkpoint not found: {path}"}
    save_json(results, args.out)
    print(results)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained video VAEs on synthetic videos.")
    parser.add_argument("--frame-ckpt", type=str, default="checkpoints/frame_vae.pt")
    parser.add_argument("--temporal-ckpt", type=str, default="checkpoints/temporal_vae.pt")
    parser.add_argument("--num-videos", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--anomaly-prob", type=float, default=0.0, help="Use >0 to evaluate on anomaly/noisy clips.")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out", type=str, default="outputs/evaluation.json")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
