from __future__ import annotations

import argparse
from typing import List

import torch

from .data import SyntheticMovingShapesDataset, VideoConfig
from .evaluate import build_model_from_checkpoint
from .utils import get_device, save_json
from .visualize import save_keyframe_strip, save_video_strip


def select_keyframes_from_latents(latents: torch.Tensor, k: int) -> List[int]:
    """Select keyframes using latent motion magnitude.

    The score for frame t is the amount of latent change around that frame. This
    favours moments where the clip changes most, while always including the first
    and last frames for context.
    """
    # latents: [T, D]
    t = latents.shape[0]
    if k >= t:
        return list(range(t))
    scores = torch.zeros(t, device=latents.device)
    diffs = torch.norm(latents[1:] - latents[:-1], dim=1)
    scores[:-1] += diffs
    scores[1:] += diffs
    selected = {0, t - 1}
    for idx in torch.argsort(scores, descending=True).tolist():
        selected.add(int(idx))
        if len(selected) >= k:
            break
    return sorted(selected)


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    device = get_device(args.device)
    model, cfg = build_model_from_checkpoint(args.ckpt, device)
    if cfg.get("model_type") != "frame_vae":
        raise ValueError("Latent keyframe summarisation currently expects a FrameVAE checkpoint because it has one latent vector per frame.")

    dataset = SyntheticMovingShapesDataset(VideoConfig(
        num_videos=max(args.index + 1, 4),
        frames=int(cfg.get("frames", 16)),
        image_size=int(cfg.get("image_size", 32)),
        anomaly_prob=args.anomaly_prob,
        seed=args.seed,
    ))
    video = dataset[args.index]["video"].unsqueeze(0).to(device)
    latents = model.encode_video_mu(video)[0]
    indices = select_keyframes_from_latents(latents, args.num_keyframes)
    save_video_strip(video[0], f"{args.out_dir}/summary_full_strip.png", title="full video")
    save_keyframe_strip(video[0], indices, f"{args.out_dir}/latent_keyframes.png")
    save_json({"checkpoint": args.ckpt, "sample_index": args.index, "keyframes": indices}, f"{args.out_dir}/latent_keyframes.json")
    print({"keyframes": indices})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Use frame-wise VAE latents to select keyframes for a simple video summary.")
    parser.add_argument("--ckpt", type=str, default="checkpoints/frame_vae.pt")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num-keyframes", type=int, default=5)
    parser.add_argument("--anomaly-prob", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out-dir", type=str, default="outputs")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
