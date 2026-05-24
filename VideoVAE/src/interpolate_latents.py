from __future__ import annotations

import argparse

import torch

from .data import build_dataset
from .evaluate import build_model_from_checkpoint
from .utils import get_device, save_json
from .visualize import save_interpolation_grid, save_video_strip


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    model, cfg = build_model_from_checkpoint(args.ckpt, device)
    frames = int(cfg.get("frames", 16))
    image_size = int(cfg.get("image_size", 32))
    dataset = build_dataset(
        dataset=args.dataset,
        data_root=args.data_root,
        video_root=args.video_root,
        split=args.split,
        num_videos=max(args.index_b + 1, 8),
        frames=frames,
        image_size=image_size,
        input_channels=int(cfg.get("input_channels", 1)),
        clips_per_video=args.clips_per_video,
        frame_stride=args.frame_stride,
        max_videos=args.max_videos,
        seed=args.seed,
    )
    video_a = dataset[args.index_a]["video"].unsqueeze(0).to(device)
    video_b = dataset[args.index_b]["video"].unsqueeze(0).to(device)

    za = model.encode_video_mu(video_a)
    zb = model.encode_video_mu(video_b)
    decoded = []
    labels = []
    alphas = torch.linspace(0, 1, args.steps, device=device)
    for alpha in alphas:
        z = (1 - alpha) * za + alpha * zb
        clip = model.decode_video_latents(z)
        decoded.append(clip[0].cpu())
        labels.append(f"a={float(alpha):.2f}")

    save_video_strip(video_a[0], f"{args.out_dir}/interpolation_source_a.png", title="source A")
    save_video_strip(video_b[0], f"{args.out_dir}/interpolation_source_b.png", title="source B")
    save_interpolation_grid(decoded, f"{args.out_dir}/latent_interpolation.png", labels=labels, frame_index=args.frame_index)
    save_json({"checkpoint": args.ckpt, "index_a": args.index_a, "index_b": args.index_b, "steps": args.steps}, f"{args.out_dir}/latent_interpolation.json")
    print(f"Saved interpolation grid to {args.out_dir}/latent_interpolation.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interpolate between two video latents and decode intermediate clips.")
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "folder", "something-v2"])
    parser.add_argument("--data-root", type=str, default=None, help="Root containing class folders, videos, frame folders, or Something-Something labels.")
    parser.add_argument("--video-root", type=str, default=None, help="Optional video directory override for Something-Something V2.")
    parser.add_argument("--split", type=str, default="validation", help="Dataset split for Something-Something V2.")
    parser.add_argument("--ckpt", type=str, default="checkpoints/temporal_vae.pt")
    parser.add_argument("--index-a", type=int, default=0)
    parser.add_argument("--index-b", type=int, default=5)
    parser.add_argument("--steps", type=int, default=7)
    parser.add_argument("--frame-index", type=int, default=8)
    parser.add_argument("--clips-per-video", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out-dir", type=str, default="outputs")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
