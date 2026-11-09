#!/usr/bin/env python3
"""Evaluate saved Pix2Pix generator checkpoints on the *paired validation set*.

Place this file in the root of junyanz/pytorch-CycleGAN-and-pix2pix,
next to train.py, and install: pip install lpips scikit-image pandas matplotlib

Example:
    python validate_pix2pix.py \
        --dataroot ./datasets/facades --name M1_baseline \
        --model pix2pix --direction BtoA --phase val \
        --checkpoints_dir ./checkpoints --metrics_dir ./validation_results

All *numbered* checkpoints (<epoch>_net_G.pth) are evaluated by default.
'latest_net_G.pth' is deliberately ignored because its true epoch is ambiguous.
The best-SSIM checkpoint is copied to best_net_G.pth; the test split is not used.

Metric definitions:
  SSIM, PSNR: computed per RGB image in [0, 1], data_range=1.0.
  LPIPS (AlexNet): per RGB image using the original [-1, 1] tensors.
  L1: unweighted mean absolute RGB pixel error in [-1, 1] (not x100).
  Edge MAE: mean of Sobel-x and Sobel-y gradient MAEs on RGB [0, 1]
            with replicate padding and kernels divided by 8; at 256x256.
All metrics are *image-level* then uniformly averaged across validation images.

The script does not change train.py or perform early stopping. Run it after
training has produced numbered checkpoints (or at a pause between training runs).
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

try:
    import lpips
except ImportError as exc:
    raise SystemExit("Missing LPIPS: run `pip install lpips scikit-image pandas matplotlib`.") from exc

from data import create_dataset
from models import create_model
from options.test_options import TestOptions


def parse_extra_arguments():
    """Separate custom validation args from official Pix2Pix TestOptions."""
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument(
        "--metrics_dir", default="./validation_results",
        help="Root folder for CSV files, plots and previews (experiment subfolder added).",
    )
    extra_parser.add_argument(
        "--epochs", nargs="+", type=int, default=None,
        help="Optional specific numbered checkpoint epochs, e.g. --epochs 10 20 30.",
    )
    extra_parser.add_argument(
        "--preview_count", type=int, default=4,
        help="Number of fixed validation examples to save per checkpoint.",
    )
    extra_parser.add_argument(
        "--dont_copy_best", action="store_true",
        help="Do not copy the best generator to checkpoints/<name>/best_net_G.pth.",
    )
    extra, remaining = extra_parser.parse_known_args()
    # Official parser reads sys.argv rather than accepting argv directly.
    sys.argv = [sys.argv[0], *remaining]
    opt = TestOptions().parse()
    return extra, opt


def find_epochs(checkpoint_dir: Path, specified: list[int] | None) -> list[int]:
    if specified:
        epochs = sorted(set(specified))
    else:
        pattern = re.compile(r"^(\d+)_net_G\.pth$")
        epochs = sorted(
            int(match.group(1))
            for path in checkpoint_dir.glob("*_net_G.pth")
            if (match := pattern.fullmatch(path.name)) is not None
        )
    missing = [e for e in epochs if not (checkpoint_dir / f"{e}_net_G.pth").is_file()]
    if missing:
        raise FileNotFoundError(f"Missing generator checkpoint(s) for epoch(s): {missing}")
    if not epochs:
        raise FileNotFoundError(
            f"No numbered checkpoints found in {checkpoint_dir}. "
            "Train with --save_epoch_freq 10 (or --save_epoch_freq 1 for a smoke test)."
        )
    return epochs


def sobel_xy(image_01: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable, fixed Sobel filters; input shape [N, C, H, W] in [0,1]."""
    if image_01.ndim != 4:
        raise ValueError("Expected image batch [N,C,H,W].")
    channels = image_01.shape[1]
    kx = image_01.new_tensor(
        [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]
    ) / 8.0
    ky = image_01.new_tensor(
        [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]
    ) / 8.0
    wx = kx.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    wy = ky.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
    padded = F.pad(image_01, (1, 1, 1, 1), mode="replicate")
    gx = F.conv2d(padded, wx, groups=channels)
    gy = F.conv2d(padded, wy, groups=channels)
    return gx, gy


def sobel_edge_mae(fake_01: torch.Tensor, real_01: torch.Tensor) -> torch.Tensor:
    """A single scalar, 0.5*(Sobel-x MAE + Sobel-y MAE)."""
    fx, fy = sobel_xy(fake_01)
    rx, ry = sobel_xy(real_01)
    return 0.5 * (F.l1_loss(fx, rx) + F.l1_loss(fy, ry))


def tensor_to_rgb_uint8(image: torch.Tensor) -> np.ndarray:
    """Convert a single [-1,1] RGB image tensor [1,3,H,W] into HxWx3 uint8."""
    array = image[0].detach().cpu().float().clamp(-1, 1)
    array = ((array + 1.0) * 0.5).permute(1, 2, 0).numpy()
    return np.round(array * 255).astype(np.uint8)


def save_preview(folder: Path, filename: str, source, prediction, target):
    """Save a 3-panel (input | generated | ground truth) PNG."""
    pieces = [tensor_to_rgb_uint8(t) for t in (source, prediction, target)]
    if not all(piece.shape == pieces[0].shape for piece in pieces):
        raise ValueError("Preview images have mismatched shapes.")
    Image.fromarray(np.concatenate(pieces, axis=1)).save(folder / filename)


def metrics_for_sample(fake: torch.Tensor, real: torch.Tensor, lpips_model):
    """Evaluate one paired RGB example; source tensors are in [-1,1]."""
    if fake.shape != real.shape or fake.shape[0] != 1 or fake.shape[1] != 3:
        raise ValueError(f"Expected matched [1,3,H,W], got {fake.shape}, {real.shape}")
    fake_01 = ((fake + 1.0) / 2.0).clamp(0.0, 1.0)
    real_01 = ((real + 1.0) / 2.0).clamp(0.0, 1.0)
    fake_np = fake_01[0].permute(1, 2, 0).cpu().numpy()
    real_np = real_01[0].permute(1, 2, 0).cpu().numpy()
    with torch.no_grad():
        score_lpips = float(lpips_model(fake, real).mean().item())
        score_l1 = float(F.l1_loss(fake, real).item())
        score_edge = float(sobel_edge_mae(fake_01, real_01).item())
    return {
        "ssim": float(structural_similarity(real_np, fake_np, channel_axis=2, data_range=1.0)),
        "lpips": score_lpips,
        "psnr": float(peak_signal_noise_ratio(real_np, fake_np, data_range=1.0)),
        "l1": score_l1,
        "edge_mae": score_edge,
    }


def plot_curves(summary: pd.DataFrame, best_epoch: int, path: Path):
    names = [("ssim", "SSIM ↑"), ("lpips", "LPIPS ↓"),
             ("psnr", "PSNR (dB) ↑"), ("l1", "Unweighted L1 ↓"),
             ("edge_mae", "Sobel Edge MAE ↓")]
    fig, axes = plt.subplots(5, 1, figsize=(9, 15), sharex=True)
    for ax, (key, label) in zip(axes, names):
        ax.plot(summary["epoch"], summary[key], marker="o", linewidth=1.5)
        ax.axvline(best_epoch, linestyle="--", linewidth=1, label=f"Best SSIM: {best_epoch}")
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("Saved checkpoint epoch")
    fig.suptitle("Pix2Pix validation metrics (higher/lower arrows indicate preference)")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    extra, opt = parse_extra_arguments()
    if opt.model != "pix2pix":
        raise ValueError("This script requires --model pix2pix.")
    if opt.phase != "val":
        raise ValueError("Validation must use --phase val; do not select on the test set.")
    if opt.direction != "BtoA":
        raise ValueError("For CMP Facades photo generation, set --direction BtoA.")
    if opt.load_iter:
        raise ValueError("Only numbered epoch checkpoints are supported. Do not pass --load_iter.")
    if extra.preview_count < 0:
        raise ValueError("--preview_count must be >= 0.")

    # Enforce *deterministic, aligned* validation preprocessing.
    # No cropping or flipping; paired real/fake comparisons use the same resize.
    opt.preprocess = "resize"
    opt.load_size = 256
    opt.crop_size = 256
    opt.no_flip = True
    opt.serial_batches = True
    opt.batch_size = 1
    opt.num_threads = 0
    opt.max_dataset_size = float("inf")
    opt.dataset_mode = "aligned"
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    root = Path(opt.dataroot)
    validation_dir = root / "val"
    if not validation_dir.is_dir():
        raise FileNotFoundError(
            f"Validation folder missing: {validation_dir}. "
            "Aligned Pix2Pix requires paired AB images under dataroot/val/."
        )

    checkpoint_dir = Path(opt.checkpoints_dir) / opt.name
    epochs = find_epochs(checkpoint_dir, extra.epochs)
    output_dir = Path(extra.metrics_dir) / opt.name
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nDevice: {opt.device} | Experiment: {opt.name}")
    print(f"Validation split: {validation_dir}")
    print("Deterministic preprocess: resize 256, no crop, no flip, fixed ordering")
    print(f"Checkpoints to evaluate: {epochs}")

    dataset = create_dataset(opt)
    if len(dataset) == 0:
        raise ValueError(f"No paired images found under {validation_dir}.")
    print(f"Validation image pairs: {len(dataset)}")

    # Pix2Pix test mode loads only netG. Initialize once then switch checkpoints.
    opt.epoch = str(epochs[0])
    model = create_model(opt)
    model.setup(opt)
    model.eval()  # Disable generator dropout; BatchNorm uses stored running stats.

    lpips_model = lpips.LPIPS(net="alex").to(opt.device).eval()
    for param in lpips_model.parameters():
        param.requires_grad_(False)

    rows = []
    for idx, epoch in enumerate(epochs):
        if idx:
            model.load_networks(str(epoch))
            model.eval()
        preview_dir = output_dir / "previews" / f"epoch_{epoch:03d}"
        if extra.preview_count:
            preview_dir.mkdir(parents=True, exist_ok=True)
        epoch_records = []
        for image_index, batch in enumerate(dataset):
            model.set_input(batch)
            with torch.inference_mode():
                model.test()
                values = metrics_for_sample(model.fake_B, model.real_B, lpips_model)
            filename = Path(model.get_image_paths()[0]).name
            record = {"epoch": epoch, "image": filename, **values}
            rows.append(record)
            epoch_records.append(record)
            if image_index < extra.preview_count:
                save_preview(
                    preview_dir, f"{image_index:03d}_{Path(filename).stem}.png",
                    model.real_A, model.fake_B, model.real_B,
                )
        mean_metrics = pd.DataFrame(epoch_records)[["ssim", "lpips", "psnr", "l1", "edge_mae"]].mean()
        print(f"epoch {epoch:4d} | " + " | ".join(f"{metric}: {value:.5f}" for metric, value in mean_metrics.items()))

    per_image = pd.DataFrame(rows)
    summary = (per_image.groupby("epoch", as_index=False, sort=True)
               .agg(n_images=("image", "size"), ssim=("ssim", "mean"),
                    lpips=("lpips", "mean"), psnr=("psnr", "mean"),
                    l1=("l1", "mean"), edge_mae=("edge_mae", "mean")))

    # Tie-break on smaller epoch; select using *validation SSIM only*.
    # Uses strict mean SSIM (not a combined weighted metric).
    best_row = summary.sort_values(["ssim", "epoch"], ascending=[False, True]).iloc[0]
    best_epoch = int(best_row["epoch"])
    best_path = checkpoint_dir / f"{best_epoch}_net_G.pth"
    copied_path = None
    if not extra.dont_copy_best:
        copied_path = checkpoint_dir / "best_net_G.pth"
        shutil.copy2(best_path, copied_path)

    per_image.to_csv(output_dir / "per_image_metrics.csv", index=False)
    summary.to_csv(output_dir / "checkpoint_metrics.csv", index=False)
    plot_curves(summary, best_epoch, output_dir / "validation_curves.png")
    best_record = {
        "experiment": opt.name,
        "selection_metric": "highest mean validation SSIM; ties favor earlier epoch",
        "epoch": best_epoch,
        "validation_image_count": int(best_row["n_images"]),
        "metrics": {key: float(best_row[key]) for key in ("ssim", "lpips", "psnr", "l1", "edge_mae")},
        "source_checkpoint": str(best_path),
        "copied_checkpoint": str(copied_path) if copied_path is not None else None,
        "validation_preprocessing": "resize_256_no_crop_no_flip_RGB_batch1",
        "test_data_used": False,
    }
    (output_dir / "best_checkpoint.json").write_text(json.dumps(best_record, indent=2), encoding="utf-8")

    print("\n=== VALIDATION COMPLETE ===")
    print(f"Best SSIM checkpoint: epoch {best_epoch} (SSIM {best_row['ssim']:.5f})")
    print(f"Metric CSV files and plots: {output_dir}")
    if copied_path:
        print(f"Best generator: {copied_path}")
        print("Test later with: python test.py ... --phase test --epoch best --eval")
    print("Do not use the test split to select checkpoints.")


if __name__ == "__main__":
    main()
