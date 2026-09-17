"""Unit and integration tests for Hướng 1 (Anatomical Inclusion Loss) and Hướng 4 (D-CMFE)."""
import math
import tempfile
import unittest
from pathlib import Path

import torch
from ml_collections import ConfigDict
from torch.utils.data import DataLoader

from training.loss.anatomical_loss import (
    AnatomicalSegmentationLoss,
    ClinicalAnatomicalInclusionLoss,
)
from training.loss.losses import SegmentationLoss
from training.models import build_model, model_from_config
from training.models.baseline_dcmfe import BaselineCMFE, BaselineDCMFE
from training.models.cmspa_net import get_testing
from training.models.modules.dcmfe import (
    CMFE_Fusion,
    DCMFE_Fusion,
    DeformableCrossModalFusion,
)
from training.trainer.trainer import run_epoch


class TestClinicalAnatomicalInclusionLoss(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_forward_output_and_gradients_canonical(self):
        loss_fn = ClinicalAnatomicalInclusionLoss(
            myo_idx=1,
            edema_idx=2,
            scar_idx=3,
            alpha=0.1,
            beta=0.05,
            label_order="canonical",
        )
        pred_logits = torch.randn(2, 4, 16, 16, requires_grad=True)
        target = torch.randint(0, 4, (2, 16, 16))

        out = loss_fn(pred_logits, target)
        self.assertIn("loss_inc", out)
        self.assertIn("loss_se", out)
        self.assertIn("loss_leak", out)
        self.assertIn("loss_scar_out", out)
        self.assertIn("loss_edema_cov", out)

        for key, val in out.items():
            self.assertTrue(torch.isfinite(val).all(), f"{key} is not finite: {val}")

        out["loss_inc"].backward()
        self.assertIsNotNone(pred_logits.grad)
        self.assertTrue(torch.isfinite(pred_logits.grad).all())
        self.assertGreater(float(pred_logits.grad.abs().sum()), 0)

    def test_forward_legacy_label_order(self):
        loss_fn = ClinicalAnatomicalInclusionLoss(label_order="legacy")
        self.assertEqual(loss_fn.scar_idx, 2)
        self.assertEqual(loss_fn.edema_idx, 3)

        pred_logits = torch.randn(2, 4, 16, 16, requires_grad=True)
        target = torch.randint(0, 4, (2, 16, 16))
        out = loss_fn(pred_logits, target)
        self.assertTrue(torch.isfinite(out["loss_inc"]))

    def test_empty_scar_no_division_by_zero(self):
        loss_fn = ClinicalAnatomicalInclusionLoss(label_order="canonical")
        pred_logits = torch.randn(2, 4, 16, 16, requires_grad=True)
        # Target contains only background and normal myocardium, no scar (3) or edema (2)
        target = torch.randint(0, 2, (2, 16, 16))
        out = loss_fn(pred_logits, target)
        self.assertTrue(torch.isfinite(out["loss_inc"]))
        self.assertEqual(float(out["loss_edema_cov"]), 0.0)

    def test_perfect_prediction_does_not_penalize_scar(self):
        loss_fn = ClinicalAnatomicalInclusionLoss(label_order="canonical")
        # Target with bg(0), myo(1), edema(2), scar(3)
        target = torch.tensor([[[0, 1, 2, 3]]], dtype=torch.long)
        logits = torch.zeros(1, 4, 1, 4)
        logits[0, 0, 0, 0] = 10.0
        logits[0, 1, 0, 1] = 10.0
        logits[0, 2, 0, 2] = 10.0
        logits[0, 3, 0, 3] = 10.0
        out = loss_fn(logits, target)
        # Perfectly predicting scar on scar must not incur large penalty
        self.assertLess(float(out["loss_scar_out"]), 1e-3)
        self.assertLess(float(out["loss_edema_cov"]), 1e-3)
        self.assertLess(float(out["loss_inc"]), 1e-3)

    def test_anatomical_segmentation_compound_loss(self):
        compound = AnatomicalSegmentationLoss(n_classes=4, alpha=0.1, beta=0.05, label_order="canonical")
        pred_logits = torch.randn(2, 4, 16, 16, requires_grad=True)
        target = torch.randint(0, 4, (2, 16, 16))

        result = compound(pred_logits, target)
        self.assertIn("loss", result)
        self.assertIn("ce", result)
        self.assertIn("dice_loss", result)
        self.assertIn("l_inc", result)

        total_check = 0.5 * result["ce"] + 0.5 * result["dice_loss"] + result["l_inc"]
        torch.testing.assert_close(result["loss"], total_check)

        result["loss"].backward()
        self.assertIsNotNone(pred_logits.grad)
        self.assertTrue(torch.isfinite(pred_logits.grad).all())


class TestDCMFEModules(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_deformable_cross_modal_fusion_pairwise(self):
        channels = 64
        module = DeformableCrossModalFusion(channels=channels, max_offset=2.0)
        feat_m = torch.randn(2, channels, 16, 16, requires_grad=True)
        feat_other = torch.randn(2, channels, 16, 16, requires_grad=True)

        aligned, offset = module.deform_align(feat_m, feat_other)
        self.assertEqual(aligned.shape, (2, channels, 16, 16))
        self.assertEqual(offset.shape, (2, 2, 16, 16))
        # Offset must be bounded by max_offset
        self.assertLessEqual(float(offset.detach().abs().max()), 2.0 + 1e-5)


        fused = module(feat_m, feat_other)
        self.assertEqual(fused.shape, (2, channels, 16, 16))

        fused.sum().backward()
        self.assertIsNotNone(feat_m.grad)
        self.assertIsNotNone(feat_other.grad)
        self.assertTrue(torch.isfinite(feat_m.grad).all())
        self.assertTrue(torch.isfinite(feat_other.grad).all())

    def test_dcmfe_fusion_3modalities(self):
        in_channels, out_channels = 64, 32
        fusion = DCMFE_Fusion(in_channels=in_channels, out_channels=out_channels, max_offset=2.0)
        cine = torch.randn(2, in_channels, 16, 16, requires_grad=True)
        psir = torch.randn(2, in_channels, 16, 16, requires_grad=True)
        t2w = torch.randn(2, in_channels, 16, 16, requires_grad=True)

        out = fusion(cine, psir, t2w)
        self.assertEqual(out.shape, (2, out_channels, 16, 16))

        out.sum().backward()
        for t in (cine, psir, t2w):
            self.assertIsNotNone(t.grad)
            self.assertTrue(torch.isfinite(t.grad).all())

    def test_cmfe_fusion_baseline(self):
        in_channels, out_channels = 64, 32
        cmfe = CMFE_Fusion(in_channels=in_channels, out_channels=out_channels)
        cine = torch.randn(2, in_channels, 8, 8, requires_grad=True)
        psir = torch.randn(2, in_channels, 8, 8, requires_grad=True)
        t2w = torch.randn(2, in_channels, 8, 8, requires_grad=True)

        out = cmfe(cine, psir, t2w)
        self.assertEqual(out.shape, (2, out_channels, 8, 8))
        out.sum().backward()
        self.assertTrue(torch.isfinite(cine.grad).all())


class TestBaselineDCMFEModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_baseline_dcmfe_forward_and_checkpoint(self):
        config = get_testing()
        config.architecture = "baseline_dcmfe"
        config.max_offset = 2.0
        config.use_sspanet = False

        model = build_model("baseline_dcmfe", config=config, img_size=32)
        self.assertIsInstance(model, BaselineDCMFE)

        images = [torch.randn(2, 1, 32, 32) for _ in range(3)]
        target = torch.randint(0, 4, (2, 32, 32))

        logits = model(*images)
        self.assertEqual(logits.shape, (2, 4, 32, 32))

        criterion = AnatomicalSegmentationLoss(n_classes=4, alpha=0.1, beta=0.05)
        loss_dict = criterion(logits, target)
        self.assertTrue(torch.isfinite(loss_dict["loss"]))
        loss_dict["loss"].backward()

        # Check that D-CMFE fusion parameters receive gradients
        for param in model.cross_fusion.parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad)
                self.assertTrue(torch.isfinite(param.grad).all())

        # Check save and load roundtrip
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_path = Path(tmpdir) / "dcmfe_test.pth"
            torch.save({"model": model.state_dict(), "model_config": model.config.to_dict()}, ckpt_path)
            loaded = torch.load(ckpt_path, weights_only=True)
            clone = model_from_config(ConfigDict(loaded["model_config"]), img_size=32)
            clone.load_state_dict(loaded["model"], strict=True)
            clone.eval()
            model.eval()
            with torch.no_grad():
                torch.testing.assert_close(model(*images), clone(*images))

    def test_baseline_cmfe_forward(self):
        config = get_testing()
        config.architecture = "baseline_cmfe"
        model = build_model("baseline_cmfe", config=config, img_size=32)
        self.assertIsInstance(model, BaselineCMFE)

        images = [torch.randn(2, 1, 32, 32) for _ in range(3)]
        logits = model(*images)
        self.assertEqual(logits.shape, (2, 4, 32, 32))

    def test_trainer_run_epoch_step_with_dcmfe_and_anatomical(self):
        config = get_testing()
        config.architecture = "baseline_dcmfe"
        model = build_model("baseline_dcmfe", config=config, img_size=32)
        device = torch.device("cpu")

        criterion = AnatomicalSegmentationLoss(n_classes=4, alpha=0.1, beta=0.05)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

        batch = {
            "image": torch.randn(2, 1, 32, 32),
            "image1": torch.randn(2, 1, 32, 32),
            "image2": torch.randn(2, 1, 32, 32),
            "label": torch.randint(0, 4, (2, 32, 32)),
        }
        loader = DataLoader([batch, batch], batch_size=None)

        metrics, step = run_epoch(
            model=model,
            loader=loader,
            criterion=criterion,
            device=device,
            amp_dtype=None,
            optimizer=optimizer,
            global_step=0,
        )

        self.assertIn("loss", metrics)
        self.assertIn("ce", metrics)
        self.assertIn("dice_loss", metrics)
        self.assertIn("l_inc", metrics)
        self.assertGreater(step, 0)
        self.assertTrue(math.isfinite(metrics["loss"]))


if __name__ == "__main__":
    unittest.main()
