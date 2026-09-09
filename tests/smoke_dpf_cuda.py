"""Explicit production-width GPU smoke test on synthetic data; no dataset training."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from training.models.cmspa_net import CMSPANet, get_config
from training.models.m3_dpf import M3DPF
from training.loss.dpf_loss import DPFLoss


def main():
    torch.set_num_threads(2)
    baseline = CMSPANet(get_config())
    baseline_count = sum(p.numel() for p in baseline.parameters())
    del baseline
    model = M3DPF(get_config()).cuda()
    print('GPU:', torch.cuda.get_device_name())
    print('M3 parameters:', baseline_count)
    print('DPF parameters:', sum(p.numel() for p in model.parameters()))
    images = [torch.randn(2, 1, 128, 128, device='cuda') for _ in range(3)]
    target = torch.randint(0, 4, (2, 128, 128), device='cuda')
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler('cuda', enabled=dtype == torch.float16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    with torch.autocast(device_type='cuda', dtype=dtype):
        output = model(*images, return_aux=True)
        losses = DPFLoss()(output, target)
    scaler.scale(losses['loss']).backward()
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert torch.isfinite(norm), 'Non-finite gradient'
    scaler.step(optimizer)
    scaler.update()
    print('AMP:', dtype, 'loss:', float(losses['loss'].detach()), 'grad norm:', float(norm))
    print('logits:', tuple(output['logits'].shape), 'aux:', tuple(output['aux_logits'].shape))
    print('peak allocated MiB:', torch.cuda.max_memory_allocated() / 2**20)


if __name__ == '__main__':
    main()
