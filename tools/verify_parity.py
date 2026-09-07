"""Compare independent SCAR/I_MMSeg imports, outputs, losses and every gradient.

Each repository runs in its own subprocess to prevent the shared `training`
package name from silently comparing aliases of one implementation. Temporary
artifacts are removed after comparison. CPU float32, fixed weights/inputs and
disabled dropout make the default tolerance meaningful.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def worker(args):
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    import torch
    import training.models.cmspa_net as implementation
    from training.loss import SegmentationLoss

    loaded = Path(implementation.__file__).resolve()
    if not loaded.is_relative_to(repo):
        raise RuntimeError(f"Wrong repository imported: {loaded}; expected {repo}")
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(123)
    config = implementation.get_testing() if args.profile == "testing" else implementation.get_config()
    config.transformer.dropout_rate = 0.0
    model = implementation.CMSPANet(config, ablation=args.ablation).train()
    shared = Path(args.artifacts) / "shared.pt"
    reference = Path(args.artifacts) / "reference.pt"
    if args.worker == "reference":
        inputs = [torch.randn(args.batch_size, 1, args.image_size, args.image_size) for _ in range(3)]
        target = torch.randint(4, (args.batch_size, args.image_size, args.image_size))
        torch.save({"state": model.state_dict(), "inputs": inputs, "target": target}, shared)
    else:
        initial = torch.load(shared, map_location="cpu", weights_only=True)
        model.load_state_dict(initial["state"], strict=True)
        inputs, target = initial["inputs"], initial["target"]
        del initial
    inputs = [value.requires_grad_() for value in inputs]
    logits = model(*inputs)
    losses = SegmentationLoss()(logits, target)
    losses["loss"].backward()
    gradients = {}
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise AssertionError(f"Disconnected/nonfinite gradient: {name}")
        gradients[name] = parameter.grad
    actual = {"logits": logits.detach(), "losses": {k: v.detach() for k, v in losses.items()},
              "input_gradients": [value.grad for value in inputs], "gradients": gradients}
    if args.worker == "reference":
        torch.save(actual, reference)
        print(json.dumps({"reference_module": str(loaded), "parameters": sum(p.numel() for p in model.parameters())}))
        return
    expected = torch.load(reference, map_location="cpu", weights_only=True)

    def compare(left, right, label):
        if not torch.isfinite(left).all() or not torch.isfinite(right).all():
            raise AssertionError(f"Nonfinite tensor in {label}")
        error = (left - right).abs().max().item()
        if error >= args.atol:
            raise AssertionError(f"{label}: max absolute error {error} >= {args.atol}")
        return error

    if set(gradients) != set(expected["gradients"]):
        raise AssertionError("Parameter names differ between repositories")
    maxima = {
        "logits": compare(actual["logits"], expected["logits"], "logits"),
        "losses": max(compare(value, expected["losses"][name], name) for name, value in actual["losses"].items()),
        "input_gradients": max(compare(a, b, "input gradient") for a, b in zip(actual["input_gradients"], expected["input_gradients"])),
        "parameter_gradients": max(compare(value, expected["gradients"][name], name) for name, value in gradients.items()),
    }
    print(json.dumps({"ablation": args.ablation, "profile": args.profile,
                      "shape": list(logits.shape), "target_module": str(loaded),
                      "gradient_tensors": len(gradients), "max_absolute_errors": maxima, "passed": True}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT.parent / "I_MMSeg")
    parser.add_argument("--profile", choices=("testing", "production"), default="testing")
    parser.add_argument("--ablations", nargs="+", choices=("M0", "M1", "M2", "M3"), default=["M0", "M1", "M2", "M3"])
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--worker", choices=("reference", "target"), help=argparse.SUPPRESS)
    parser.add_argument("--repo", help=argparse.SUPPRESS)
    parser.add_argument("--ablation", help=argparse.SUPPRESS)
    parser.add_argument("--artifacts", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.image_size < 32 or args.image_size % 16 or min(args.batch_size, args.threads, args.atol) <= 0:
        parser.error("Use image-size >=32 divisible by 16 and positive batch-size, threads and atol")
    if args.worker:
        worker(args)
        return
    source = args.source_root.resolve()
    if source == ROOT or not (source / "training/models/cmspa_net.py").is_file():
        parser.error("source-root must point to a separate I_MMSeg repository")
    for ablation in args.ablations:
        with tempfile.TemporaryDirectory(prefix="scar_parity_") as artifacts:
            for role, repo in (("reference", source), ("target", ROOT)):
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", role,
                           "--repo", str(repo), "--ablation", ablation, "--artifacts", artifacts,
                           "--profile", args.profile, "--image-size", str(args.image_size),
                           "--batch-size", str(args.batch_size), "--threads", str(args.threads),
                           "--atol", str(args.atol)]
                subprocess.run(command, check=True, cwd=repo)


if __name__ == "__main__":
    main()
