#!/usr/bin/env python3
"""Evaluate the validation-selected M1/M2/M3 Pix2Pix generators on held-out Facades test pairs.

Reuses metrics_for_sample from validate_pix2pix.py to keep metric definitions identical.
Loads ONLY <experiment>/best_net_G.pth; never selects checkpoints using test data.

Run from the repository root, e.g.:
 python evaluate_test.py --dataroot ./datasets/facades \
   --name M1_baseline --model pix2pix --direction BtoA --phase test \
   --checkpoints_dir /content/drive/MyDrive/pix2pix_checkpoints \
   --metrics_dir /content/drive/MyDrive/pix2pix_test_results \
   --names M1_baseline M2_single_edge M3_multiscale_edge
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lpips
import numpy as np
import pandas as pd
import torch

from data import create_dataset
from models import create_model
from options.test_options import TestOptions
try:
    from validate_pix2pix import metrics_for_sample, save_preview
except ImportError as exc:
    raise SystemExit(
        "Place validate_pix2pix.py alongside evaluate_test.py in the repo root. "
        "Both scripts must use exactly the same metric definitions."
    ) from exc

METRIC_KEYS = ("ssim", "lpips", "psnr", "l1", "edge_mae")


def parse_options():
    # Strip script-only arguments before invoking the official Pix2Pix option parser.
    extra_parser = argparse.ArgumentParser(add_help=False)
    extra_parser.add_argument("--names", nargs="+", required=True,
                              help="Experiment checkpoint folders, e.g. M1_baseline M2_single_edge M3_multiscale_edge")
    extra_parser.add_argument("--metrics_dir", default="./test_results")
    extra_parser.add_argument("--validation_dir", default=None,
                              help="Optional directory containing <experiment>/best_checkpoint.json, for reporting selected epochs.")
    extra_parser.add_argument("--preview_count", type=int, default=5)
    extra, remaining = extra_parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    opt = TestOptions().parse()
    return extra, opt


def main():
    extra, opt = parse_options()
    if opt.model != "pix2pix" or opt.direction != "BtoA" or opt.phase != "test":
        raise ValueError("Use --model pix2pix --direction BtoA --phase test for this experiment.")
    if extra.preview_count < 0:
        raise ValueError("--preview_count must be nonnegative.")
    if len(set(extra.names)) != len(extra.names):
        raise ValueError("Experiment names must be unique.")

    # Exactly match the deterministic preprocessing of validate_pix2pix.py.
    opt.dataset_mode = "aligned"
    opt.preprocess = "resize"
    opt.load_size = 256
    opt.crop_size = 256
    opt.no_flip = True
    opt.serial_batches = True
    opt.batch_size = 1
    opt.num_threads = 0
    opt.max_dataset_size = float("inf")
    opt.epoch = "best"
    # Official test.py creates this explicitly; TestOptions does not supply it.
    opt.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if opt.load_iter:
        raise ValueError("Do not use --load_iter; the evaluator requires best_net_G.pth.")

    test_dir = Path(opt.dataroot) / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"Missing held-out test split: {test_dir}")

    ckpt_root = Path(opt.checkpoints_dir)
    for experiment in extra.names:
        path = ckpt_root / experiment / "best_net_G.pth"
        if not path.is_file():
            raise FileNotFoundError(
                f"Best generator not found: {path}\n"
                "Run validate_pix2pix.py on the VAL split first so it copies best_net_G.pth."
            )

    dataset = create_dataset(opt)
    if len(dataset) == 0:
        raise ValueError(f"No paired test images found in {test_dir}")
    print(f"\nHeld-out test images: {len(dataset)}")
    print("Identical preprocessing for all models: resize 256, no crop/flip, sorted order.")
    print("Checkpoint selection: already completed using validation SSIM (NOT test data).")

    lpips_model = lpips.LPIPS(net="alex").to(opt.device).eval()
    for p in lpips_model.parameters():
        p.requires_grad_(False)

    output_dir = Path(extra.metrics_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    all_image_rows = []

    for experiment in extra.names:
        opt.name = experiment
        opt.epoch = "best"
        model = create_model(opt)
        model.setup(opt)  # Loads checkpoints/<experiment>/best_net_G.pth
        model.eval()

        preview_dir = output_dir / experiment / "previews"
        if extra.preview_count:
            preview_dir.mkdir(parents=True, exist_ok=True)

        records = []
        for i, batch in enumerate(dataset):
            model.set_input(batch)
            with torch.inference_mode():
                model.test()
                scores = metrics_for_sample(model.fake_B, model.real_B, lpips_model)

            filename = Path(model.get_image_paths()[0]).name
            record = {"model": experiment, "image": filename, **scores}
            records.append(record)
            all_image_rows.append(record)

            if i < extra.preview_count:
                save_preview(
                    preview_dir,
                    f"{i:03d}_{Path(filename).stem}.png",
                    model.real_A, model.fake_B, model.real_B,
                )

        # Best epoch is informational only; never used to choose a checkpoint from test data.
        selected_epoch = None
        if extra.validation_dir:
            best_metadata = Path(extra.validation_dir) / experiment / "best_checkpoint.json"
            if best_metadata.is_file():
                selected_epoch = json.loads(best_metadata.read_text(encoding="utf-8")).get("epoch")

        summary = {
            "model": experiment,
            "selected_validation_epoch": selected_epoch,
            "test_images": len(records),
            **{key: float(np.mean([r[key] for r in records])) for key in METRIC_KEYS},
        }
        rows.append(summary)
        print(f"\n{experiment}: " + "  ".join(f"{key}={summary[key]:.5f}" for key in METRIC_KEYS))

    summary_df = pd.DataFrame(rows)
    image_df = pd.DataFrame(all_image_rows)
    summary_df.to_csv(output_dir / "test_comparison.csv", index=False)
    image_df.to_csv(output_dir / "test_per_image_metrics.csv", index=False)

    print("\n=== HELD-OUT TEST RESULTS ===")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.5f}"))
    print(f"\nSaved: {output_dir / 'test_comparison.csv'}")
    print(f"Saved: {output_dir / 'test_per_image_metrics.csv'}")
    print("Do not use these test results to retune the loss weights or select checkpoints.")


if __name__ == "__main__":
    main()
