from __future__ import annotations

import torch

from src.data import SyntheticMovingShapesDataset, VideoConfig
from src.losses import reconstruction_metrics
from src.models import FrameVAE, TemporalVAE
from src.summarize_video import select_keyframes_from_latents
from src.utils import ensure_dir, save_checkpoint, save_json, set_seed
from src.visualize import save_interpolation_grid, save_keyframe_strip, save_reconstruction_comparison, save_video_strip


def main() -> None:
    """Fast smoke test for the full project structure.

    This does not train meaningful models. It runs forward passes with randomly
    initialised VAEs, writes compatible checkpoints, and saves visual outputs so
    that you can confirm the codebase is installed correctly. Use the training
    commands in the README for actual experiments.
    """
    set_seed(42)
    ensure_dir("checkpoints")
    ensure_dir("outputs")

    dataset = SyntheticMovingShapesDataset(VideoConfig(num_videos=8, frames=16, image_size=32, seed=42))
    video_a = dataset[0]["video"].unsqueeze(0)
    video_b = dataset[5]["video"].unsqueeze(0)

    frame_vae = FrameVAE(latent_dim=16)
    temporal_vae = TemporalVAE(latent_dim=32)

    with torch.no_grad():
        frame_recon, _, _ = frame_vae(video_a)
        temporal_recon, _, _ = temporal_vae(video_a)
        metrics = {
            "frame_vae_untrained": reconstruction_metrics(frame_recon, video_a),
            "temporal_vae_untrained": reconstruction_metrics(temporal_recon, video_a),
        }

        # Save compatible checkpoints so that the other scripts can be tested.
        save_checkpoint(
            "checkpoints/frame_vae.pt",
            frame_vae,
            config={"model_type": "frame_vae", "latent_dim": 16, "frames": 16, "image_size": 32, "input_channels": 1},
        )
        save_checkpoint(
            "checkpoints/temporal_vae.pt",
            temporal_vae,
            config={"model_type": "temporal_vae", "latent_dim": 32, "frames": 16, "image_size": 32, "input_channels": 1},
        )

        save_video_strip(video_a[0], "outputs/input_strip.png", title="input video")
        save_reconstruction_comparison(video_a[0], temporal_recon[0], "outputs/reconstruction_comparison.png")

        za = temporal_vae.encode_video_mu(video_a)
        zb = temporal_vae.encode_video_mu(video_b)
        decoded = []
        labels = []
        for alpha in torch.linspace(0, 1, 5):
            z = (1 - alpha) * za + alpha * zb
            decoded.append(temporal_vae.decode_video_latents(z)[0])
            labels.append(f"a={float(alpha):.2f}")
        save_interpolation_grid(decoded, "outputs/latent_interpolation.png", labels=labels)

        frame_latents = frame_vae.encode_video_mu(video_a)[0]
        keyframes = select_keyframes_from_latents(frame_latents, k=5)
        save_keyframe_strip(video_a[0], keyframes, "outputs/latent_keyframes.png")
        save_json({"metrics": metrics, "keyframes": keyframes}, "outputs/demo_summary.json")

    print("Demo complete. This was a fast smoke test using untrained models.")
    print("For meaningful results, run the training commands in README.md.")


if __name__ == "__main__":
    main()
