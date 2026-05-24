from __future__ import annotations

import argparse

import torch

from .data import build_dataset
from .evaluate import build_model_from_checkpoint, reconstruct_deterministic
from .losses import reconstruction_metrics
from .utils import get_device, save_json
from .visualize import save_reconstruction_comparison, save_video_strip


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    model, cfg = build_model_from_checkpoint(args.ckpt, device)
    dataset = build_dataset(
        dataset=args.dataset,
        data_root=args.data_root,
        video_root=args.video_root,
        split=args.split,
        num_videos=max(args.index + 1, 4),
        frames=int(cfg.get("frames", 16)),
        image_size=int(cfg.get("image_size", 32)),
        input_channels=int(cfg.get("input_channels", 1)),
        anomaly_prob=args.anomaly_prob,
        clips_per_video=args.clips_per_video,
        frame_stride=args.frame_stride,
        max_videos=args.max_videos,
        seed=args.seed,
    )
    sample = dataset[args.index]
    video = sample["video"].unsqueeze(0).to(device)
    recon = reconstruct_deterministic(model, video)
    metrics = reconstruction_metrics(recon, video)

    save_video_strip(video[0], f"{args.out_dir}/input_strip.png", title="input video")
    save_video_strip(recon[0], f"{args.out_dir}/reconstruction_strip.png", title="reconstructed video")
    save_reconstruction_comparison(video[0], recon[0], f"{args.out_dir}/reconstruction_comparison.png")
    save_json({"checkpoint": args.ckpt, "sample_index": args.index, "metrics": metrics}, f"{args.out_dir}/reconstruction_metrics.json")
    print(metrics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Reconstruct one synthetic or folder-based video clip and save visualisations.")
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "folder", "something-v2"])
    parser.add_argument("--data-root", type=str, default=None, help="Root containing class folders, videos, frame folders, or Something-Something labels.")
    parser.add_argument("--video-root", type=str, default=None, help="Optional video directory override for Something-Something V2.")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split for Something-Something V2.")
    parser.add_argument("--ckpt", type=str, default="checkpoints/temporal_vae.pt")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=999)
    parser.add_argument("--anomaly-prob", type=float, default=0.0)
    parser.add_argument("--clips-per-video", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out-dir", type=str, default="outputs")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
