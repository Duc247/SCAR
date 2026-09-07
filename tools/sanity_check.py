"""One real forward/backward/AdamW update without changing production defaults.

Defaults to full production M3, batch 1 at 128x128 on CPU. Pass --profile testing
for the smaller structural fixture. Inputs here are synthetic; use
tools/smoke_pipeline.py for a complete small real-data epoch.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from training.models.cmspa_net import CMSPANet, get_config, get_testing
from training.loss import SegmentationLoss
from training.trainer.trainer import resolve_amp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("production", "testing"), default="production")
    parser.add_argument("--ablation", choices=("M0", "M1", "M2", "M3"), default="M3")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--amp", choices=("none", "auto", "fp16", "bf16"), default="none")
    args = parser.parse_args()
    if args.image_size < 32 or args.image_size % 16 or min(args.batch_size, args.threads) <= 0:
        parser.error("image-size must be >=32 and divisible by 16; batch-size/threads must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(1234)
    config = get_testing() if args.profile == "testing" else get_config()
    model = CMSPANet(config, ablation=args.ablation).to(args.device).train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4, foreach=False)
    amp = resolve_amp(args.amp, torch.device(args.device))
    scaler = torch.amp.GradScaler("cuda", enabled=amp == torch.float16)
    inputs = [torch.randn(args.batch_size, 1, args.image_size, args.image_size, device=args.device) for _ in range(3)]
    target = torch.randint(4, (args.batch_size, args.image_size, args.image_size), device=args.device)
    before = model.segmentation_head[0].weight.detach().clone()
    with torch.autocast(args.device, dtype=amp, enabled=amp is not None):
        logits = model(*inputs)
        losses = SegmentationLoss()(logits, target)
    scaler.scale(losses["loss"]).backward()
    scaler.unscale_(optimizer)
    for name, parameter in model.named_parameters():
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise AssertionError(f"Disconnected/nonfinite gradient: {name}")
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    scaler.step(optimizer)
    scaler.update()
    if not all(torch.isfinite(parameter).all() for parameter in model.parameters()):
        raise AssertionError("Optimizer produced nonfinite parameters")
    if torch.equal(before, model.segmentation_head[0].weight):
        raise AssertionError("AdamW failed to update the segmentation head")
    print(json.dumps({"passed": True, "data": "synthetic", "profile": args.profile,
                      "ablation": args.ablation, "device": args.device, "shape": list(logits.shape),
                      "parameters": sum(p.numel() for p in model.parameters()),
                      "amp_dtype": str(amp),
                      "peak_allocated_mb": torch.cuda.max_memory_allocated() / 2**20 if args.device == "cuda" else None,
                      "losses": {k: float(v.detach()) for k, v in losses.items()},
                      "gradient_norm_before_clip": float(norm)}, indent=2))


if __name__ == "__main__":
    main()
