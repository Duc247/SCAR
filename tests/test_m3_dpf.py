import csv
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from ml_collections import ConfigDict
from torch.utils.data import DataLoader

from training.models import model_from_config
from training.models.cmspa_net import CMSPANet, get_testing
from training.models.m3_dpf import M3DPF
from training.loss.dpf_loss import DPFLoss, present_dice
from training.loss.losses import SegmentationLoss
from training.metrics.confusion_meter import ConfusionMeter
from training.metrics.surface_distance import benchmark_rows, summarize_rows
from training.train import parse_args, main
from training.trainer.trainer import append_metrics_csv, run_epoch


class DPFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_forward_gradients_checkpoint_and_legacy(self):
        torch.manual_seed(12)
        model = M3DPF(get_testing(), img_size=32)
        images = [torch.randn(2, 1, 32, 32) for _ in range(3)]
        target = torch.randint(0, 4, (2, 32, 32))
        output = model(*images, return_aux=True)
        self.assertEqual(output['aux_logits'].shape, (2, 2, 8, 8))
        loss = DPFLoss()(output, target)
        loss['loss'].backward()
        for fusion in (model.cross_fusion, model.feature_fusion[1]):
            self.assertIsNotNone(fusion.eta_logit.grad)
            for router in fusion.routers:
                grad = router[-1].weight.grad
                self.assertTrue(torch.isfinite(grad).all())
                self.assertGreater(float(grad.abs().sum()), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pth'
            torch.save({'model': model.state_dict(), 'model_config': model.config.to_dict()}, path)
            checkpoint = torch.load(path, weights_only=True)
        clone = model_from_config(ConfigDict(checkpoint['model_config']), img_size=32)
        clone.load_state_dict(checkpoint['model'], strict=True)
        model.eval(); clone.eval()
        with torch.no_grad():
            torch.testing.assert_close(model(*images), clone(*images))
            torch.testing.assert_close(model(*images), model(*images, return_aux=True)['logits'])
        legacy_config = get_testing()
        torch.manual_seed(17)
        direct = CMSPANet(legacy_config)
        torch.manual_seed(17)
        factory = model_from_config(legacy_config)
        self.assertEqual(set(direct.state_dict()), set(factory.state_dict()))
        for key, value in direct.state_dict().items():
            torch.testing.assert_close(value, factory.state_dict()[key])
        self.assertNotIn('architecture', legacy_config)

    def test_losses_empty_and_perfect(self):
        perfect = torch.ones(2, 4, 4)
        self.assertAlmostEqual(float(present_dice(perfect, perfect)), 0)
        for class_id in (0, 2, 3):
            target = torch.full((2, 16, 16), class_id)
            logits = torch.randn(2, 4, 16, 16, requires_grad=True)
            aux = torch.randn(2, 2, 4, 4, requires_grad=True)
            criterion = DPFLoss()
            result = criterion({'logits': logits, 'aux_logits': aux}, target)
            self.assertTrue(all(torch.isfinite(v) for v in result.values()))
            result['loss'].backward()
            self.assertTrue(torch.isfinite(logits.grad).all())
            self.assertTrue(torch.isfinite(aux.grad).all())
            criterion.set_epoch(0)
            zero = criterion({'logits': logits, 'aux_logits': aux}, target)
            torch.testing.assert_close(zero['loss'], .5 * zero['ce'] + .5 * zero['dice_loss'])

    def test_precision_recall_and_empty_cases(self):
        # Scar: TP=1, FP=2, FN=1; deliberately unequal precision and recall.
        truth = torch.tensor([3, 3, 0, 0, 0])
        pred = torch.tensor([3, 0, 3, 3, 0])
        meter = ConfusionMeter(); meter.update(pred, truth)
        scores = meter.compute()
        self.assertAlmostEqual(scores['precision/scar'], 1 / 3)
        self.assertAlmostEqual(scores['recall/scar'], 1 / 2)
        self.assertTrue(math.isnan(scores['recall/edema']))
        rows = benchmark_rows(pred.numpy().reshape(1, 1, 5), truth.numpy().reshape(1, 1, 5), 'x', False)
        summary = summarize_rows(rows)
        self.assertAlmostEqual(summary['scar']['mean_precision'], 1 / 3)
        self.assertEqual(summary['edema']['recall_undefined_cases'], 1)
        missed = benchmark_rows(np.zeros((1, 1, 5)), truth.numpy().reshape(1, 1, 5), 'y', False)
        score = summarize_rows(missed)['scar']
        self.assertIsNone(score['mean_precision'])
        self.assertEqual(score['mean_recall'], 0)

    def test_training_validation_and_csv_upgrade(self):
        model = M3DPF(get_testing(), img_size=32)
        samples = [{**{key: torch.randn(1, 32, 32) for key in ('image', 'image1', 'image2')},
                    'label': torch.randint(0, 4, (32, 32))} for _ in range(3)]
        loader = DataLoader(samples, batch_size=2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        train, step = run_epoch(model, loader, DPFLoss(), torch.device('cpu'), None,
                                optimizer=optimizer, accum_steps=2)
        self.assertEqual(step, 1)
        self.assertIn('aux_loss', train)
        val, _ = run_epoch(model, loader, DPFLoss(), torch.device('cpu'), None)
        self.assertIn('precision/scar', val)
        self.assertIn('recall/edema', val)
        legacy, _ = run_epoch(CMSPANet(get_testing()), loader, SegmentationLoss(), torch.device('cpu'), None)
        self.assertNotIn('aux_loss', legacy)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'metrics.csv'
            append_metrics_csv(path, {'epoch': 1, 'loss': .5})
            append_metrics_csv(path, {'epoch': 2, 'loss': .4, 'precision': .7})
            append_metrics_csv(path, {'epoch': 3, 'loss': .3, 'precision': .8})
            with path.open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(rows[0]['precision'], '')
            self.assertEqual(rows[2]['precision'], '0.8')
            self.assertTrue(all(None not in row for row in rows))

    def test_config_is_opt_in(self):
        args, config = parse_args(['--config', 'training/config/models/m3_dpf.yaml'])
        self.assertEqual(config['model']['architecture'], 'm3_dpf')
        self.assertEqual(args.patience, 35)
        args, config = parse_args([])
        self.assertNotIn('architecture', config['model'])
        self.assertEqual(args.patience, 0)

    def test_custom_dpf_yaml_reaches_trainer(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'custom.yaml'
            path.write_text('model:\n  architecture: m3_dpf\n  dpf_bottleneck_width: 12\n'
                            '  dpf_skip_width: 8\n  dpf_loss_version: 1\n', encoding='utf-8')
            with patch('training.train.Trainer') as trainer:
                main(['--config', str(path), '--data-root', directory, '--list-dir', directory,
                      '--output-dir', str(Path(directory) / 'run')])
                model = trainer.call_args.args[0]
                self.assertEqual(model.cross_fusion.projections[0][0].out_channels, 12)
                self.assertEqual(model.feature_fusion[1].projections[0][0].out_channels, 8)
                self.assertEqual(model.config.dpf_loss_version, 1)

    def test_dpf_rejects_invalid_contract(self):
        for value in (0, -1, 1.5, True):
            config = get_testing()
            config.dpf_skip_width = value
            with self.assertRaisesRegex(ValueError, 'positive integer'):
                M3DPF(config)
        with self.assertRaisesRegex(ValueError, 'four canonical'):
            M3DPF(get_testing(), num_classes=3)
        model = M3DPF(get_testing(), ablation=None)
        self.assertEqual(model.ablation, 'M3')


if __name__ == '__main__':
    unittest.main()
