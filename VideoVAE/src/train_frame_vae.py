from __future__ import annotations

import argparse
from typing import Dict

import torch
from tqdm import tqdm

from .data import build_dataloaders
from .losses import reconstruction_metrics, vae_loss
from .models import FrameVAE
from .utils import get_device, save_checkpoint, save_json, set_seed


def train(args: argparse.Namespace) -> Dict[str, float]:
    set_seed(args.seed)
    device = get_device(args.device)
    train_loader, val_loader = build_dataloaders(
        dataset=args.dataset,
        data_root=args.data_root,
        video_root=args.video_root,
        split=args.split,
        num_videos=args.num_videos,
        frames=args.frames,
        image_size=args.image_size,
        input_channels=args.input_channels,
        batch_size=args.batch_size,
        anomaly_prob=0.0,
        clips_per_video=args.clips_per_video,
        frame_stride=args.frame_stride,
        max_videos=args.max_videos,
        seed=args.seed,
        val_fraction=args.val_fraction,
    )
    model = FrameVAE(latent_dim=args.latent_dim, image_size=args.image_size, input_channels=args.input_channels).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = {"loss": 0.0, "recon": 0.0, "kl": 0.0}
        pbar = tqdm(train_loader, desc=f"FrameVAE epoch {epoch}/{args.epochs}", leave=False)
        for batch in pbar:
            video = batch["video"].to(device)
            recon, mu, logvar = model(video)
            losses = vae_loss(recon, video, mu, logvar, beta=args.beta)
            opt.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            for k in running:
                running[k] += float(losses[k].detach().cpu())
            pbar.set_postfix(loss=float(losses["loss"].detach().cpu()))

        train_summary = {f"train_{k}": v / max(1, len(train_loader)) for k, v in running.items()}
        val_summary = validate(model, val_loader, device, args.beta)
        epoch_summary = {"epoch": epoch, **train_summary, **val_summary}
        history.append(epoch_summary)
        print(epoch_summary)

    config = vars(args).copy()
    config["model_type"] = "frame_vae"
    save_checkpoint(args.out, model, config=config, extra={"history": history})
    save_json({"history": history, "config": config}, args.metrics_out)
    return history[-1] if history else {}


@torch.no_grad()
def validate(model: FrameVAE, loader, device: torch.device, beta: float) -> Dict[str, float]:
    model.eval()
    totals = {"val_loss": 0.0, "val_recon": 0.0, "val_kl": 0.0, "val_mse": 0.0, "val_psnr": 0.0, "val_ssim_global": 0.0, "val_temporal_mse": 0.0}
    for batch in loader:
        video = batch["video"].to(device)
        recon, mu, logvar = model(video)
        losses = vae_loss(recon, video, mu, logvar, beta=beta)
        metrics = reconstruction_metrics(recon, video)
        totals["val_loss"] += float(losses["loss"].cpu())
        totals["val_recon"] += float(losses["recon"].cpu())
        totals["val_kl"] += float(losses["kl"].cpu())
        totals["val_mse"] += metrics["mse"]
        totals["val_psnr"] += metrics["psnr"]
        totals["val_ssim_global"] += metrics["ssim_global"]
        totals["val_temporal_mse"] += metrics["temporal_mse"]
    return {k: v / max(1, len(loader)) for k, v in totals.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a frame-wise VAE baseline on synthetic or folder-based videos.")
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "folder", "something-v2"])
    parser.add_argument("--data-root", type=str, default=None, help="Root containing class folders, videos, frame folders, or Something-Something labels.")
    parser.add_argument("--video-root", type=str, default=None, help="Optional video directory override for Something-Something V2.")
    parser.add_argument("--split", type=str, default="train", help="Dataset split for Something-Something V2.")
    parser.add_argument("--num-videos", type=int, default=1000)
    parser.add_argument("--frames", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--input-channels", type=int, default=1, choices=[1, 3])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--clips-per-video", type=int, default=1)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--out", type=str, default="checkpoints/frame_vae.pt")
    parser.add_argument("--metrics-out", type=str, default="outputs/frame_vae_metrics.json")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
