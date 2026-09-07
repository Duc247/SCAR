"""Integration checks for optimization, resumability and CLI evaluation."""
import copy
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from training.dataset.data_contract import CANONICAL_LABEL_ORDER
from training.train import main as train_main
from training.trainer.trainer import load_checkpoint, run_epoch
from training.loss import SegmentationLoss


class TinyDataset(Dataset):
    def __init__(self, count=5):
        self.x = torch.randn(count, 1, 8, 8, generator=torch.Generator().manual_seed(3))
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        x = self.x[index]
        return dict(image=x, image1=x, image2=x, label=(x[0] > 0).long())


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
    def forward(self, a, b, c):
        return self.conv(torch.cat([a, b, c], 1))


def test_epoch_accumulation_uneven_microbatches_matches_large_batch():
    torch.manual_seed(1)
    a, data = TinyModel(), TinyDataset()
    b = copy.deepcopy(a)
    for model, batch_size, accumulation in ((a, 5, 1), (b, 2, 3)):
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
        result, step = run_epoch(model, DataLoader(data, batch_size=batch_size), SegmentationLoss(),
            torch.device('cpu'), None, optimizer, torch.amp.GradScaler('cuda', enabled=False), scheduler,
            accum_steps=accumulation, clip_grad=0)
        assert step == 1 and result['optimizer_updates'] == 1 and result['samples'] == 5
    for pa, pb in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(pa, pb, rtol=1e-5, atol=1e-7)


def test_short_training_reduces_tiny_objective():
    torch.manual_seed(5)
    model, data = TinyModel(), TinyDataset()
    loader = DataLoader(data, batch_size=5)
    criterion = SegmentationLoss()
    before, _ = run_epoch(model, loader, criterion, torch.device('cpu'), None)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    for _ in range(12):
        run_epoch(model, loader, criterion, torch.device('cpu'), None, optimizer,
                  torch.amp.GradScaler('cuda', enabled=False), scheduler)
    after, _ = run_epoch(model, loader, criterion, torch.device('cpu'), None)
    assert after['loss'] < before['loss'] * .7


def create_fixture(root):
    data, lists = root / 'data', root / 'lists'
    lists.mkdir(parents=True)
    names = {'train': ['case01_slice000', 'case01_slice001', 'case02_slice000'],
             'val': ['case03_slice000'], 'val_vol': ['case03'], 'test_vol': ['case04']}
    for split, entries in names.items():
        (lists / f'{split}.txt').write_text('\n'.join(entries) + '\n')
    random = np.random.default_rng(8)
    labels = np.zeros((32, 32), dtype=np.uint8)
    labels[4:28, 4:28] = 1
    labels[8:16, 8:16] = 2
    labels[16:24, 16:24] = 3
    for modality in ('bSSFP', 'LGE', 'T2w'):
        slices, volumes = data / modality / 'train_npz', data / modality / 'test_vol_h5'
        slices.mkdir(parents=True)
        volumes.mkdir(parents=True)
        for name in names['train'] + names['val']:
            np.savez(slices / f'{name}.npz', image=random.random((32, 32)).astype(np.float32),
                     label=labels, label_order=CANONICAL_LABEL_ORDER)
        with h5py.File(volumes / 'case04.npy.h5', 'w') as stream:
            stream['image'] = random.random((32, 32, 3)).astype(np.float32)
            stream['label'] = np.repeat(labels[:, :, None], 3, axis=2)
            stream.attrs['label_order'] = CANONICAL_LABEL_ORDER
            stream.attrs['spacing_unit'] = 'mm'
            stream['spacing'] = [1., 1., 2.]
            stream['affine'] = np.diag([1., 1., 2., 1.])
    return data, lists


def test_train_resume_exact_and_evaluate(tmp_path):
    data, lists = create_fixture(tmp_path)
    base = ['--data-root', str(data), '--list-dir', str(lists), '--config', 'testing',
            '--img-size', '32', '--epochs', '2', '--batch-size', '2', '--accum-steps', '2',
            '--device', 'cpu', '--amp', 'none', '--cpu-threads', '2', '--log-every', '1']
    direct, resumed = tmp_path / 'direct', tmp_path / 'resumed'
    train_main(base + ['--output-dir', str(direct)])
    train_main(base + ['--output-dir', str(resumed), '--epochs-per-run', '1'])
    first = load_checkpoint(resumed / 'last.pth')
    assert first['global_step'] == 1
    del first
    train_main(base + ['--output-dir', str(resumed), '--resume', str(resumed / 'last.pth')])
    a, b = load_checkpoint(direct / 'last.pth'), load_checkpoint(resumed / 'last.pth')
    assert a['global_step'] == b['global_step'] == 2
    assert a['scheduler'] == b['scheduler']
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], atol=0, rtol=0)
    assert (resumed / 'best.pth').exists()
    lines = [json.loads(line) for line in (resumed / 'metrics.jsonl').read_text().splitlines()]
    assert [line['epoch'] for line in lines] == [1, 2]
    assert lines[-1]['lr_next'] == 0
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    for folder, tag in [('epochs', 'val/mean_dice'), ('updates', 'step/loss')]:
        events = EventAccumulator(str(resumed / 'tensorboard' / folder))
        events.Reload()
        assert [event.step for event in events.Scalars(tag)] == [1, 2]
    assert {'train/ce', 'val/dice_loss', 'val/mean_iou', 'epoch_seconds', 'checkpoint_seconds'} <= lines[-1].keys()
    import training.evaluate as module
    metrics = module.main(['--checkpoint', str(resumed / 'best.pth'), '--data-root', str(data),
                           '--device', 'cpu', '--amp', 'none', '--cpu-threads', '2', '--batch-size', '2'])
    assert metrics['case_count'] == 1
    assert (resumed / 'evaluation_test_vol' / 'case04_pred.nii.gz').exists()
    assert 'edema_inclusive' in metrics and 'myocardial_ring' in metrics
    assert metrics['scar']['hd95_mm_defined_cases'] + metrics['scar']['hd95_mm_undefined_cases'] == 1
    bad_lists = tmp_path / 'leaking_lists'
    bad_lists.mkdir()
    (bad_lists / 'test_vol.txt').write_text('case03\n')
    with pytest.raises(ValueError, match='model selection'):
        module.main(['--checkpoint', str(resumed / 'best.pth'), '--data-root', str(data),
                     '--list-dir', str(bad_lists), '--device', 'cpu'])
    (resumed / 'splits' / 'train.txt').write_text('case_altered_slice000\n')
    with pytest.raises(ValueError, match='modified after training'):
        module.main(['--checkpoint', str(resumed / 'best.pth'), '--data-root', str(data), '--device', 'cpu'])


def test_early_stopping_uses_cumulative_gain_and_saves_true_best(tmp_path, monkeypatch):
    import training.trainer.trainer as trainer
    from ml_collections import ConfigDict
    from training.train import build_parser
    data, lists = create_fixture(tmp_path)
    directory = tmp_path / 'early_stop'
    args = build_parser().parse_args(['--data-root', str(data), '--list-dir', str(lists),
        '--output-dir', str(directory), '--epochs', '8', '--patience', '2', '--min-delta', '.001',
        '--device', 'cpu', '--amp', 'none', '--num-workers', '0', '--no-tensorboard'])
    model = TinyModel()
    model.config = ConfigDict({'test': True})
    scores = iter([.5, .5006, .5012, .5018, .5017, .5015])
    def fake_epoch(model, loader, criterion, device, amp_dtype, optimizer=None, *pos, **kw):
        score = 0. if optimizer is not None else next(scores)
        return dict(loss=1., mean_dice=score, mean_iou=score, gpu_peak_allocated_mb=0.), 1
    saved = []
    monkeypatch.setattr(trainer, 'run_epoch', fake_epoch)
    monkeypatch.setattr(trainer, 'atomic_checkpoint', lambda path, payload: saved.append((Path(path).name,
        payload['epoch'], payload['best_score'], payload['bad_epochs'], payload['patience_score'])))
    trainer.trainer_Myops(args, model, directory)
    last = [row for row in saved if row[0] == 'last.pth'][-1]
    assert last == ('last.pth', 4, .5018, 2, .5012)
    assert [row for row in saved if row[0] == 'best.pth'][-1][1] == 3


def test_config_precedence_and_invalid_paths(tmp_path):
    from training.train import parse_args
    args, config = parse_args([])
    assert args.batch_size == 16 and args.base_lr == .001
    assert config['model']['resnet']['num_layers'] == [3, 4, 9]
    args, _ = parse_args(['--config', 'testing', '--batch-size', '3'])
    assert args.batch_size == 3 and args.img_size == 32
    with pytest.raises(FileNotFoundError):
        parse_args(['--config', str(tmp_path / 'missing/cmspa_net.yaml')])
    bad = tmp_path / 'bad.yaml'
    bad.write_text('train:\n  batch_szie: 8\n')
    with pytest.raises(ValueError, match='Unknown configuration'):
        parse_args(['--config', str(bad)])
    bad.write_text('amp: wrong\n')
    with pytest.raises(ValueError, match='Invalid amp'):
        parse_args(['--config', str(bad)])


def test_one_click_raw_cache_train_evaluate_and_nifti(tmp_path):
    import nibabel as nib
    from test_data import raw_fixture, write_lists
    from run_all import main as pipeline_main
    from preprocessing.build_splits import build_splits
    from training.predict import main as predict_main
    source, _, _, affine = raw_fixture(tmp_path)
    fixed = tmp_path / 'held/test_vol.txt'
    write_lists(fixed.parent, test_vol=['case0005'])
    data, lists = tmp_path / 'cache', tmp_path / 'lists'
    run = pipeline_main(['--config', 'training/config/models/testing.yaml', '--run-id', 'e2e',
        '--run-root', str(tmp_path / 'runs'), '--data-root', str(data), '--raw-root', str(source),
        '--list-dir', str(lists), '--test-list', str(fixed), '--epochs', '1', '--batch-size', '2',
        '--num-workers', '0', '--cpu-threads', '2', '--device', 'cpu', '--amp', 'none', '--no-tensorboard'])
    assert (run / 'best.pth').is_file() and (run / 'last.pth').is_file()
    assert (run / 'evaluation_test_vol/per_case.csv').is_file()
    result = json.loads((run / 'evaluation_test_vol/metrics.json').read_text())
    assert result['case_count'] == 1
    # Imported cache must retain its original patient partition.
    build_splits(data, lists, test_list=fixed)
    prediction = tmp_path / 'prediction.nii.gz'
    argv = ['--checkpoint', str(run / 'best.pth'), '--cine', str(source / 'bSSFP/case0005.nii.gz'),
        '--lge', str(source / 'LGE/case0005.nii.gz'), '--t2w', str(source / 'T2w/case0005.nii.gz'),
        '--output', str(prediction), '--normalization', 'unit255', '--device', 'cpu', '--cpu-threads', '2']
    predict_main(argv)
    volume = nib.load(prediction)
    np.testing.assert_array_equal(volume.affine, affine)
    assert volume.header.get_xyzt_units()[0] == 'mm' and volume.shape == (8, 8, 2)
    with pytest.raises(FileExistsError):
        predict_main(argv)
    # Remove only the fixture's physical metadata, then exercise unknown-unit evaluation.
    for modality in ('bSSFP', 'LGE', 'T2w'):
        path = data / modality / 'test_vol_h5/case0005.npy.h5'
        with h5py.File(path, 'a') as stream:
            del stream['spacing']
            del stream['affine']
            stream.attrs['spacing_unit'] = 'unknown'
    from training.evaluate import main as evaluate_main
    metrics = evaluate_main(['--checkpoint', str(run / 'best.pth'), '--data-root', str(data),
        '--device', 'cpu', '--cpu-threads', '2', '--no-save-predictions', '--output-dir', str(tmp_path / 'unknown')])
    for name in ('normal_myocardium', 'edema', 'scar', 'edema_inclusive', 'myocardial_ring'):
        assert metrics[name]['hd95_mm_defined_mean'] is None
        assert metrics[name]['hd95_mm_defined_cases'] == 0
