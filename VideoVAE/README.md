# Video VAE for Latent Video Compression and Generation

<i>How much temporal information can a VAE preserve while compressing video into a low-dimensional latent space?</i>


This project is a lightweight research-style implementation of a **Video Variational Autoencoder (Video VAE)** pipeline, inspired by [this paper by P. Wu et. al](https://arxiv.org/pdf/2411.06449). It explores how much temporal information a VAE can preserve when compressing short video clips into low-dimensional latent representations.

The project is intentionally runnable without downloading large video datasets. By default, it uses a synthetic moving-shapes dataset.


The project compares two models:

1. **Frame-wise VAE**  
   Compresses each frame independently using 2D convolutions. This is a baseline that captures spatial appearance but does not explicitly model temporal dynamics.

2. **Temporal Video VAE**  
   Compresses the entire video clip using 3D convolutions over time, height, and width. This model learns a single latent vector for a full clip, so it must preserve both spatial and motion information.

---

## What this project demonstrates

This project demonstrates:

- Video latent representation learning
- Frame-wise vs temporal VAE comparison
- Video reconstruction
- Latent-space interpolation between clips
- Simple latent-based video summarisation
- Reconstruction metrics such as MSE, PSNR, SSIM approximation, and temporal reconstruction error
- A small but complete pipeline that can be extended to real datasets

---

## Setup

Create and activate a virtual environment.

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Quick demo

Run:

```bash
python run_demo.py
```

This performs a fast smoke-test run using randomly initialised models. It verifies that the dataset, models, metrics, visualisation utilities, checkpoint format, reconstruction visualisation, latent interpolation, and keyframe summarisation code all execute correctly.

The outputs will be saved in:

```text
outputs/
```

Important: the demo is only meant to confirm that the codebase works. The generated reconstructions are not meaningful because the demo does not train the models. For meaningful results, train using the commands below.

---

## Full training workflow

### Step 1: Train the frame-wise VAE baseline

```bash
python -m src.train_frame_vae \
  --num-videos 2000 \
  --epochs 30 \
  --batch-size 32 \
  --latent-dim 32 \
  --out checkpoints/frame_vae.pt \
  --metrics-out outputs/frame_vae_metrics.json
```

This trains a VAE that treats each frame independently.

Conceptually:

```text
frame -> 2D Conv Encoder -> latent vector -> 2D Conv Decoder -> reconstructed frame
```

Since every frame is processed independently, this model may reconstruct individual frames reasonably well, but it does not directly encode motion as a sequence.

---

### Step 2: Train the temporal video VAE

```bash
python -m src.train_temporal_vae \
  --num-videos 2000 \
  --epochs 40 \
  --batch-size 32 \
  --latent-dim 64 \
  --out checkpoints/temporal_vae.pt \
  --metrics-out outputs/temporal_vae_metrics.json
```

This trains a VAE that compresses the full video clip using 3D convolutions.

Conceptually:

```text
video clip -> 3D Conv Encoder -> latent vector -> 3D Conv Decoder -> reconstructed video clip
```

This model is more suitable for video because the encoder sees time, height, and width together.

---

### Step 3: Evaluate both models

```bash
python -m src.evaluate \
  --frame-ckpt checkpoints/frame_vae.pt \
  --temporal-ckpt checkpoints/temporal_vae.pt \
  --num-videos 300 \
  --out outputs/evaluation.json
```

The evaluation reports:

| Metric | Meaning | Better |
|---|---|---|
| MSE | Pixel-wise reconstruction error | Lower |
| PSNR | Signal-to-noise reconstruction quality | Higher |
| SSIM global | Dependency-free global SSIM approximation | Higher |
| Temporal MSE | Error in frame-to-frame motion differences | Lower |

The most relevant metric for this project is **temporal MSE**, because it checks whether the reconstruction preserves motion, not just frame appearance.

---

### Step 4: Reconstruct a video clip

```bash
python -m src.reconstruct_clip \
  --ckpt checkpoints/temporal_vae.pt \
  --index 0 \
  --out-dir outputs
```

This saves:

```text
outputs/input_strip.png
outputs/reconstruction_strip.png
outputs/reconstruction_comparison.png
outputs/reconstruction_metrics.json
```

The comparison image shows:

```text
input video
reconstructed video
absolute error
```

---

### Step 5: Interpolate between two video latents

```bash
python -m src.interpolate_latents \
  --ckpt checkpoints/temporal_vae.pt \
  --index-a 0 \
  --index-b 5 \
  --steps 7 \
  --out-dir outputs
```

This encodes two videos into latent vectors, linearly interpolates between them, decodes the intermediate latents, and saves:

```text
outputs/latent_interpolation.png
outputs/interpolation_source_a.png
outputs/interpolation_source_b.png
```

This demonstrates that the VAE latent space can be used for simple generative operations.

---

### Step 6: Latent-based video summarisation

```bash
python -m src.summarize_video \
  --ckpt checkpoints/frame_vae.pt \
  --index 0 \
  --num-keyframes 5 \
  --out-dir outputs
```

This uses the frame-wise VAE latents to select keyframes. Frames with larger latent changes are treated as more informative.

Output:

```text
outputs/latent_keyframes.png
outputs/latent_keyframes.json
outputs/summary_full_strip.png
```

This is a simple demonstration of how learned latents can support downstream tasks beyond reconstruction.

---

## Dataset used by default

The default dataset is synthetic and generated on the fly. Each clip contains a moving object:

- square
- circle
- diamond

Each clip has shape:

```text
[channels, frames, height, width] = [1, 16, 32, 32]
```

The object moves across the frame and bounces off boundaries. The dataset can also inject anomalies such as sudden jumps or occluded frames.

You can evaluate on anomalous clips using:

```bash
python -m src.evaluate \
  --frame-ckpt checkpoints/frame_vae.pt \
  --temporal-ckpt checkpoints/temporal_vae.pt \
  --anomaly-prob 0.5 \
  --out outputs/evaluation_anomalous.json
```

---

## How to extend this to real video datasets

To use UCF101, Kinetics, or Something-Something V2, replace `SyntheticMovingShapesDataset` in `src/data.py` with a dataset class that returns the same format:

```python
{
    "video": FloatTensor with shape [1, T, H, W] or [3, T, H, W],
    "label": class_label,
    "is_anomaly": 0 or 1
}
```

For RGB videos, update the model input channels from `1` to `3` in `src/models.py`.

Recommended preprocessing for real datasets:

1. Decode videos into frames.
2. Resize frames to a fixed size, for example 64x64 or 128x128.
3. Sample fixed-length clips, for example 16 or 32 frames.
4. Normalize pixel values to `[0, 1]`.
5. Train the same FrameVAE and TemporalVAE models.
