from __future__ import annotations

import argparse

import torch

from .data import SyntheticMovingShapesDataset, VideoConfig
from .evaluate import build_model_from_checkpoint
from .losses import reconstruction_metrics
from .utils import get_device, save_json
from .visualize import save_reconstruction_comparison, save_video_strip


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    model, cfg = build_model_from_checkpoint(args.ckpt, device)
    dataset = SyntheticMovingShapesDataset(VideoConfig(
        num_videos=max(args.index + 1, 4),
        frames=int(cfg.get("frames", 16)),
        image_size=int(cfg.get("image_size", 32)),
        anomaly_prob=args.anomaly_prob,
        seed=args.seed,
    ))
    sample = dataset[args.index]
    video = sample["video"].unsqueeze(0).to(device)
    recon, _, _ = model(video)
    metrics = reconstruction_metrics(recon, video)

    save_video_strip(video[0], f"{args.out_dir}/input_strip.png", title="input video")
    save_video_strip(recon[0], f"{args.out_dir}/reconstruction_strip.png", title="reconstructed video")
    save_reconstruction_comparison(video[0], recon[0], f"{args.out_dir}/reconstruction_comparison.png")
    save_json({"checkpoint": args.ckpt, "sample_index": args.index, "metrics": metrics}, f"{args.out_dir}/reconstruction_metrics.json")
    print(metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct one synthetic video clip and save visualisations.")
    parser.add_argument("--ckpt", type=str, default="checkpoints/temporal_vae.pt")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--anomaly-prob", type=float, default=0.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out-dir", type=str, default="outputs")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
