"""Evaluate a saved prompt-free checkpoint on held-out 3D cases."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

from ml_collections import ConfigDict
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.dataset.data_contract import CLASS_NAMES, patient_id, read_split_names
from training.dataset.myops_dataset import MyopsDataset, Myops_dataset
from training.predict import predict_volume
from training.metrics.surface_distance import binary_metrics
from training.models.cmspa_net import CMSPANet, VisionTransformer
from training.trainer.trainer import (
    json_safe,
    load_checkpoint,
    resolve_amp,
    resolve_device,
    write_json,
)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--list-dir", default=None, help="defaults to saved run splits")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split", choices=["test_vol", "val_vol"], default="test_vol")
    parser.add_argument("--batch-size", type=int, default=2, help="inference slices per batch")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", choices=["auto", "none", "fp16", "bf16"], default="auto")
    parser.add_argument(
        "--label-order",
        choices=["auto", "legacy", "canonical"],
        default=None,
        help="default: checkpoint data convention",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        nargs=3,
        default=None,
        metavar=("H_MM", "W_MM", "D_MM"),
        help="explicit physical spacing for legacy files missing geometry",
    )
    parser.add_argument("--save-predictions", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow-voxel-spacing", action="store_true",
                        help="Report voxel HD95 separately when physical geometry is unknown")
    parser.add_argument("--cpu-threads", type=int, default=4)
    return parser


def summarize_rows(rows):
    summary = {}
    for name in sorted({row["region"] for row in rows}):
        region_rows = [r for r in rows if r["region"] == name]
        region = {
            "cases": len(region_rows),
            "status_counts": dict(Counter(r["status"] for r in region_rows)),
        }
        for metric in ("dice", "iou"):
            values = [r[metric] for r in region_rows if r[metric] is not None]
            region[f"mean_{metric}"] = float(np.mean(values)) if values else None
            region[f"{metric}_defined_cases"] = len(values)
        for unit in ("mm", "voxel"):
            values = [r[f"hd95_{unit}"] for r in region_rows if r[f"hd95_{unit}"] is not None]
            region[f"hd95_{unit}_defined_mean"] = float(np.mean(values)) if values else None
            region[f"hd95_{unit}_defined_cases"] = len(values)
            region[f"hd95_{unit}_undefined_cases"] = len(region_rows) - len(values)
            asd_values = [r[f"asd_{unit}"] for r in region_rows if r.get(f"asd_{unit}") is not None]
            region[f"asd_{unit}_defined_mean"] = float(np.mean(asd_values)) if asd_values else None
            region[f"asd_{unit}_defined_cases"] = len(asd_values)
            region[f"asd_{unit}_undefined_cases"] = len(region_rows) - len(asd_values)
        summary[name] = region
    for metric in ("dice", "iou"):
        values = [summary[name][f"mean_{metric}"] for name in CLASS_NAMES[1:] if summary[name][f"mean_{metric}"] is not None]
        summary[f"macro_foreground_{metric}"] = float(np.mean(values)) if values else None
    return summary


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.batch_size < 1 or args.cpu_threads < 1:
        raise ValueError("batch-size and cpu-threads must be positive")
    if args.spacing is not None and (not np.isfinite(args.spacing).all() or min(args.spacing) <= 0):
        raise ValueError("Spacing must be finite and positive.")
    torch.set_num_threads(args.cpu_threads)
    checkpoint = load_checkpoint(args.checkpoint)
    device = resolve_device(args.device)
    amp_dtype = resolve_amp(args.amp, device)

    config = ConfigDict(checkpoint["model_config"])
    model = CMSPANet(config, img_size=checkpoint["args"]["img_size"], num_classes=4)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()

    checkpoint_path = Path(args.checkpoint).resolve()
    run_dir = checkpoint_path.parent
    if run_dir.name == "checkpoints" or (not (run_dir / "splits").exists() and (run_dir.parent / "splits").exists()):
        run_dir = run_dir.parent

    list_dir = Path(args.list_dir) if args.list_dir else run_dir / "splits"
    output = Path(args.output_dir) if args.output_dir else run_dir / ("evaluation_" + args.split)
    output.mkdir(parents=True, exist_ok=True)
    roots = [str(Path(args.data_root) / modality / f"{args.split}_h5") for modality in ("bSSFP", "LGE", "T2w")]

    for split_name, expected in checkpoint["split_hashes"].items():
        manifest = run_dir / "splits" / f"{split_name}.txt"
        if hashlib.sha256(manifest.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Saved {split_name} manifest was modified after training.")

    training_ids = {patient_id(name) for name in read_split_names(run_dir / "splits", "train")}
    validation_ids = {patient_id(name) for name in read_split_names(run_dir / "splits", "val")}
    evaluation_ids = {patient_id(name) for name in read_split_names(list_dir, args.split)}
    forbidden_ids = training_ids | validation_ids if args.split == "test_vol" else training_ids
    overlap = forbidden_ids & evaluation_ids
    if overlap:
        raise ValueError(
            f"Evaluation split overlaps patients used for training/model selection: {sorted(overlap)[:10]}"
        )

    dataset = MyopsDataset(
        *roots,
        str(list_dir),
        args.split,
        label_order=args.label_order or checkpoint["args"]["label_order"],
    )
    rows, total_seconds, total_slices = [], 0.0, 0
    for sample in dataset:
        case = sample["case_name"]
        images = [sample[k] for k in ("image", "image1", "image2")]
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        prediction = predict_volume(
            model,
            images,
            checkpoint["args"]["img_size"],
            args.batch_size,
            device,
            amp_dtype,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        total_seconds += elapsed
        total_slices += prediction.shape[2]
        target = np.asarray(sample["label"])
        known_geometry = bool(sample.get("has_geometry", False))
        spacing = (
            args.spacing
            if args.spacing is not None
            else (sample["spacing"] if known_geometry and sample.get("spacing_unit") == "mm" else None)
        )
        if spacing is not None and known_geometry and args.spacing is not None and not np.allclose(spacing, sample["spacing"]):
            raise ValueError("Explicit spacing conflicts with stored geometry.")
        if spacing is not None and sample.get("has_affine", False):
            axes = np.asarray(sample["affine"])[:3, :3]
            axes = axes / np.linalg.norm(axes, axis=0)
            if not np.allclose(axes.T @ axes, np.eye(3), atol=1e-4):
                raise ValueError(
                    "HD95 with axis spacing requires an orthogonal grid; resample sheared NIfTI upstream."
                )
        regions = [(name, prediction == i, target == i) for i, name in enumerate(CLASS_NAMES) if i]
        regions.extend([
            ("edema_inclusive", prediction >= 2, target >= 2),
            ("myocardial_ring", prediction >= 1, target >= 1),
        ])
        for name, pred, truth in regions:
            metrics = binary_metrics(pred, truth, spacing,
                                     compute_distance=spacing is not None or args.allow_voxel_spacing,
                                     empty_mode="undefined")
            unit = "mm" if spacing is not None else ("voxel" if args.allow_voxel_spacing else "unknown")
            rows.append(
                dict(
                    case=case,
                    region=name,
                    **metrics,
                    hd95_unit=unit,
                    hd95_mm=metrics["hd95"] if unit == "mm" else None,
                    hd95_voxel=metrics["hd95"] if unit == "voxel" else None,
                    hd95_status=(metrics["status"] if metrics["status"] != "ok" else
                                 ("missing_physical_geometry" if unit == "unknown" else "ok")),
                    asd_unit=unit,
                    asd_mm=metrics["asd"] if unit == "mm" else None,
                    asd_voxel=metrics["asd"] if unit == "voxel" else None,
                    asd_status=(metrics["status"] if metrics["status"] != "ok" else
                                ("missing_physical_geometry" if unit == "unknown" else "ok")),
                    inference_seconds=elapsed,
                )
            )
        if args.save_predictions:
            np.savez_compressed(
                output / f"{case}_pred.npz",
                prediction=prediction,
                class_names=np.asarray(CLASS_NAMES),
                spacing=np.asarray(spacing) if spacing is not None else np.full(3, np.nan),
                affine=np.asarray(sample["affine"]),
                spacing_unit="mm" if spacing is not None else "unknown",
            )
            if sample.get("has_affine", False):
                import nibabel as nib

                nifti = nib.Nifti1Image(prediction, np.asarray(sample["affine"]))
                if sample.get("spacing_unit") == "mm":
                    nifti.header.set_xyzt_units("mm")
                nib.save(nifti, output / f"{case}_pred.nii.gz")
        print(
            f"{case}: {prediction.shape[2]} slices, {elapsed:.3f}s, HD95 unit={unit}",
            flush=True,
        )
    if not rows:
        raise ValueError("No evaluation cases.")
    with (output / "per_case.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = summarize_rows(rows)
    summary.update(
        checkpoint=str(checkpoint_path),
        checkpoint_epoch=checkpoint["epoch"] + 1,
        ablation=config.ablation,
        split=args.split,
        case_count=len(dataset),
        inference_seconds=total_seconds,
        inference_slices_per_second=total_slices / total_seconds,
        timing_note="Includes transfer and resizing, excludes disk I/O/metrics; first case includes warmup.",
        hd95_note="mm and voxel distances are never mixed. Unknown physical geometry gives null mm HD95. Means include only defined surfaces; inspect counts. Missed lesions have Dice/IoU=0.",
        device=str(device),
        amp_dtype=str(amp_dtype),
    )
    write_json(output / "metrics.json", summary)
    print(json.dumps(json_safe(summary), indent=2))
    return summary


if __name__ == "__main__":
    main()
