# Video VAE for Latent Video Compression

A compact PyTorch research project for learning latent representations of short video clips with variational autoencoders.

The project compares two approaches:

- **FrameVAE**: compresses each frame independently with 2D convolutions.
- **TemporalVAE**: compresses an entire clip with 3D convolutions over time, height, and width.

It can run on a synthetic moving-shapes dataset, or on real videos such as Something-Something V2.

## What This Shows

This repository demonstrates:

- Video reconstruction with VAEs
- Frame-wise vs temporal latent compression
- Latent-space interpolation between clips
- Latent-based keyframe selection
- Reconstruction metrics including MSE, PSNR, approximate SSIM, and temporal MSE
- A practical pipeline for moving from synthetic clips to real video data

The central question is:

> How much visual and temporal information can a small VAE preserve when compressing short videos into low-dimensional latent vectors?

## Example Outputs

The main outputs are saved PNG strips and JSON metrics:

```text
outputs/
  reconstruction/
    input_strip.png
    reconstruction_strip.png
    reconstruction_comparison.png
    reconstruction_metrics.json
  latent_interpolation.png
  latent_keyframes.png
  evaluation.json
```

`reconstruction_comparison.png` shows:

```text
input video frames
reconstructed video frames
mean absolute error
```

The temporal VAE is intentionally small, so early results are blurry. The useful signal is whether training improves reconstruction quality and whether motion is preserved across frames.

## Project Structure

```text
video_vae_latent_video/
  src/
    data.py                  # Synthetic, folder, and Something-Something V2 datasets
    models.py                # FrameVAE and TemporalVAE
    losses.py                # VAE loss and reconstruction metrics
    train_frame_vae.py       # Frame-wise baseline training
    train_temporal_vae.py    # Temporal VAE training
    evaluate.py              # Metric evaluation
    reconstruct_clip.py      # Reconstruction visualization
    interpolate_latents.py   # Latent interpolation demo
    summarize_video.py       # Latent keyframe selection
    visualize.py             # PNG visualizations
  scripts/
    prepare_something_something_v2.ps1
  checkpoints/
  outputs/
  run_demo.py
  requirements.txt
```

The previous long-form workflow notes are kept in [`README_FULL_WORKFLOW.md`](README_FULL_WORKFLOW.md).

## Setup

Create a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If you want GPU training, install a CUDA-enabled PyTorch build that matches your driver. For example, for CUDA 13.0:

```powershell
pip install --force-reinstall --no-deps torch==2.12.0+cu130 --index-url https://download.pytorch.org/whl/cu130
```

Check CUDA:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Quick Smoke Test

Run:

```powershell
python run_demo.py
```

This creates synthetic examples and untrained model outputs. It verifies that the code runs, but the reconstructions are not meaningful yet.

## Train on Synthetic Video

Train the frame-wise baseline:

```powershell
python -m src.train_frame_vae `
  --num-videos 2000 `
  --epochs 30 `
  --batch-size 32 `
  --latent-dim 32 `
  --out checkpoints/frame_vae.pt `
  --metrics-out outputs/frame_vae_metrics.json
```

Train the temporal VAE:

```powershell
python -m src.train_temporal_vae `
  --num-videos 2000 `
  --epochs 40 `
  --batch-size 32 `
  --latent-dim 64 `
  --out checkpoints/temporal_vae.pt `
  --metrics-out outputs/temporal_vae_metrics.json
```

Evaluate both:

```powershell
python -m src.evaluate `
  --frame-ckpt checkpoints/frame_vae.pt `
  --temporal-ckpt checkpoints/temporal_vae.pt `
  --num-videos 300 `
  --out outputs/evaluation.json
```

## Train on Something-Something V2

This repository does not include the dataset. Download the Something-Something V2 archive parts and official labels separately.

Expected download files:

```text
20bn-something-something-v2-00
20bn-something-something-v2-01
20bn-something-something-download-package-labels.zip
```

Prepare labels only:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\prepare_something_something_v2.ps1
```

Extract a small training subset:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\prepare_something_something_v2.ps1 -ExtractVideos -MaxVideos 500
```

Train the temporal VAE on the extracted subset:

```powershell
python -m src.train_temporal_vae `
  --dataset something-v2 `
  --data-root data/something_something_v2 `
  --split train `
  --input-channels 3 `
  --image-size 64 `
  --frames 16 `
  --batch-size 8 `
  --epochs 50 `
  --latent-dim 128 `
  --out checkpoints/temporal_vae.pt `
  --metrics-out outputs/temporal_vae_metrics.json
```

## Visualize Reconstructions

After training:

```powershell
python -m src.reconstruct_clip `
  --dataset something-v2 `
  --data-root data/something_something_v2 `
  --split train `
  --ckpt checkpoints/temporal_vae.pt `
  --index 0 `
  --out-dir outputs/reconstruction
```

For a more informative visual check, choose a high-motion clip instead of a mostly static one:

```powershell
python -m src.reconstruct_clip `
  --dataset something-v2 `
  --data-root data/something_something_v2 `
  --split train `
  --ckpt checkpoints/temporal_vae.pt `
  --index 435 `
  --out-dir outputs/reconstruction_motion
```

## Latent Interpolation

```powershell
python -m src.interpolate_latents `
  --dataset something-v2 `
  --data-root data/something_something_v2 `
  --split train `
  --ckpt checkpoints/temporal_vae.pt `
  --index-a 0 `
  --index-b 5 `
  --steps 7 `
  --out-dir outputs
```

## Keyframe Selection

Keyframe selection uses the frame-wise VAE because it produces one latent vector per frame:

```powershell
python -m src.summarize_video `
  --dataset something-v2 `
  --data-root data/something_something_v2 `
  --split train `
  --ckpt checkpoints/frame_vae.pt `
  --index 0 `
  --num-keyframes 5 `
  --out-dir outputs
```

## Metrics

The project reports:

| Metric | Meaning | Better |
|---|---|---|
| MSE | Pixel reconstruction error | Lower |
| PSNR | Reconstruction signal-to-noise quality | Higher |
| SSIM global | Dependency-free global SSIM approximation | Higher |
| Temporal MSE | Error in frame-to-frame differences | Lower |

For video, `temporal_mse` is especially useful because it measures whether motion/change is preserved, not just whether individual frames look similar.

## Notes and Limitations

- The models are intentionally small and educational.
- Reconstructions will be blurry compared with modern video generative models.
- The temporal VAE compresses a whole clip into one latent vector, which is a hard bottleneck.
- The default real-video setup uses 64x64 clips for manageable training.
- This is not a diffusion model and does not include perceptual, adversarial, or LPIPS losses.
- Dataset archives, extracted videos, generated outputs, and checkpoints should generally not be committed to GitHub.

## Suggested `.gitignore`

```text
.venv/
__pycache__/
*.pyc
data/
checkpoints/*.pt
outputs/
```

## Resume Summary

Built a PyTorch video VAE pipeline for latent video compression and reconstruction, comparing frame-wise and temporal encoders on synthetic and real video data with reconstruction metrics, latent interpolation, and keyframe summarization.
