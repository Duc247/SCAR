"""One real-data epoch on a tiny patient-disjoint subset, without shrinking M3.

Temporary checkpoints are removed automatically. This verifies execution, not
generalization or A100 capacity; production defaults are never changed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from training.dataset.data_contract import read_split_names, patient_id, validate_patient_splits, _write_manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="E:/STUDY/DATASET/MyoPS380/Processed_data")
    parser.add_argument("--list-dir", default=str(ROOT / "data/processed/splits"))
    parser.add_argument("--raw-root", default=None, help="Also verify raw NIfTI prediction and affine preservation")
    parser.add_argument("--output", default=str(ROOT / "outputs/verification/local_epoch"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--amp", choices=("none", "auto", "fp16", "bf16"), default="none")
    parser.add_argument("--label-order", choices=("auto", "legacy", "canonical"), default="legacy")
    args = parser.parse_args(argv)
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Choose a new smoke output directory: {output}")
    all_splits = {name: read_split_names(args.list_dir, name) for name in ("train", "val", "test_vol")}
    validate_patient_splits(all_splits)
    selected = {"train": all_splits["train"][:2], "val": all_splits["val"][:1],
                "test_vol": all_splits["test_vol"][:1]}
    selected["val_vol"] = sorted({patient_id(n) for n in selected["val"]})
    with tempfile.TemporaryDirectory(prefix="scar_smoke_") as temporary:
        scratch = Path(temporary)
        lists, run = scratch / "lists", scratch / "run"
        for split, values in selected.items():
            _write_manifest(lists / f"{split}.txt", values)
        common = ["--device", args.device, "--amp", args.amp, "--cpu-threads", "2"]
        command = [sys.executable, str(ROOT / "training/train.py"),
                   "--config", str(ROOT / "training/config/models/cmspa_net.yaml"),
                   "--data-root", str(Path(args.data_root).resolve()), "--list-dir", str(lists),
                   "--output-dir", str(run), "--epochs", "1", "--batch-size", "1",
                   "--accum-steps", "1", "--num-workers", "0", "--no-tensorboard",
                   "--label-order", args.label_order, *common]
        subprocess.run(command, cwd=ROOT, check=True)
        subprocess.run([sys.executable, str(ROOT / "training/evaluate.py"),
                        "--checkpoint", str(run / "best.pth"), "--data-root", args.data_root,
                        "--batch-size", "1", "--no-save-predictions", *common], cwd=ROOT, check=True)
        nifti_verified = False
        if args.raw_root:
            import nibabel as nib
            import numpy as np
            case = selected["test_vol"][0]
            paths = [Path(args.raw_root) / m / f"{case}.nii.gz" for m in ("bSSFP", "LGE", "T2w")]
            prediction = scratch / "prediction.nii.gz"
            subprocess.run([sys.executable, str(ROOT / "training/predict.py"),
                            "--checkpoint", str(run / "best.pth"), "--cine", str(paths[0]),
                            "--lge", str(paths[1]), "--t2w", str(paths[2]), "--output", str(prediction),
                            "--normalization", "unit255", "--batch-size", "1", *common], cwd=ROOT, check=True)
            reference, actual = nib.load(paths[0]), nib.load(prediction)
            assert reference.shape == actual.shape
            np.testing.assert_allclose(reference.affine, actual.affine, rtol=0, atol=0)
            assert reference.header.get_xyzt_units() == actual.header.get_xyzt_units()
            assert set(np.unique(actual.get_fdata())) <= {0, 1, 2, 3}
            nifti_verified = True
        record = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        config = json.loads((run / "config.json").read_text(encoding="utf-8"))
        assert config["parameters"] == 64_403_442
        assert record["train/optimizer_updates"] == 2 and record["epoch"] == 1
        output.mkdir(parents=True, exist_ok=True)
        for name in ("config.json", "metrics.csv", "metrics.jsonl", "summary.json", "train.log"):
            shutil.copy2(run / name, output / name)
        shutil.copytree(run / "evaluation_test_vol", output / "evaluation_test_vol")
        report = {"status": "passed", "profile": "production M3", "parameters": config["parameters"],
                  "image_size": 128, "train_slices": 2, "validation_slices": 1, "test_patients": 1,
                  "epochs": 1, "batch_size": 1, "optimizer_updates": 2,
                  "nifti_affine_and_units_verified": nifti_verified,
                  "scope": "Execution smoke test on a small real subset; metrics are not research results.",
                  "device": args.device, "temporary_checkpoints_removed": True}
    (output / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
