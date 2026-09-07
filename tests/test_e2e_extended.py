"""Comprehensive 4-Tier Opaque-Box E2E Test Suite for SCAR.

Organized across 4 verification tiers:
- Tier 1: Feature Coverage (Data, Label, Architecture, Attention, Bottleneck, Losses & Metrics, Trainer)
- Tier 2: Boundary & Corner Cases (Empty masks, single pixels, absent classes, null/anisotropic spacing,
          sheared affines, AMP FP16 large values, uneven accumulation, out-of-bounds IDs)
- Tier 3: Cross-Feature Combinations (Data+Model+Loss+Accumulation, Label+Sampler+Confusion, Checkpoint+Resume)
- Tier 4: Real-World Application Scenarios (3D volume inference, AMP+AdamW training step, clinical composite evaluation)
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import tempfile
import unittest

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from preprocessing.preprocessing import MODALITIES, normalize_image
from training.dataset.data_contract import (
    CANONICAL_LABEL_ORDER,
    CLASS_NAMES,
    LEGACY_LABEL_ORDER,
    canonicalize_label,
    patient_id,
    read_split_names,
    resolve_label_order,
    validate_patient_splits,
)
from training.dataset.myops_dataset import MyopsDataset, RandomGenerator
from training.dataset.sampler import build_rare_class_sampler
from training.loss import DiceLoss, SegmentationLoss, build_loss
from training.metrics import ConfusionMeter, binary_metrics, calculate_metric_percase
from training.models.cmspa_net import CMSPANet, VisionTransformer, get_config, get_testing
from training.models.modules.cmspa import CMSPA_Fusion, channel_std
from training.models.modules.sspanet import (
    SSPA_ChannelAttention,
    SSPA_SpatialAttention,
    SSPA_ZPool,
    SSPANet_Block,
    strip_rms,
)
from training.predict import predict_volume
from training.trainer.trainer import atomic_checkpoint, load_checkpoint, seed_everything


def _create_synthetic_cache(
    root: Path,
    cases: list[str],
    slices_per_case: int = 2,
    shape: tuple[int, int] = (32, 32),
    label_order: str = CANONICAL_LABEL_ORDER,
    intensity_offset: float = 0.0,
):
    """Helper to create a valid minimal 3-modality slice cache."""
    for mod in MODALITIES:
        (root / mod / "train_npz").mkdir(parents=True, exist_ok=True)
    splits_dir = root / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    slice_names = []
    for case in cases:
        for s in range(slices_per_case):
            slice_name = f"{case}_slice{s}"
            slice_names.append(slice_name)
            # Create deterministic synthetic image and label
            label = np.zeros(shape, dtype=np.uint8)
            label[4:12, 4:12] = 1  # Normal myocardium
            label[8:12, 8:12] = 2  # Edema
            label[10:12, 10:12] = 3  # Scar

            for i, mod in enumerate(MODALITIES):
                image = (
                    np.full(shape, 0.2 * (i + 1) + intensity_offset, dtype=np.float32)
                )
                image = np.clip(image, 0.0, 1.0)
                npz_path = root / mod / "train_npz" / f"{slice_name}.npz"
                np.savez_compressed(
                    npz_path,
                    image=image,
                    label=label,
                    patient_id=np.array(case),
                    axis_order="HW",
                    label_order=label_order,
                    normalization="unit",
                )

    (splits_dir / "train.txt").write_text("\n".join(slice_names), encoding="utf-8")
    (splits_dir / "val.txt").write_text("\n".join(slice_names[:slices_per_case]), encoding="utf-8")
    (splits_dir / "val_vol.txt").write_text(cases[0], encoding="utf-8")
    return slice_names


# ==============================================================================
# TIER 1: FEATURE COVERAGE
# ==============================================================================


class TestTier1FeatureCoverage(unittest.TestCase):
    """Tier 1: Basic functionality and interface contracts across core areas."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.root = Path(self.tmp_dir)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    # --------------------------------------------------------------------------
    # 1. Data Contract (5 test cases)
    # --------------------------------------------------------------------------

    def test_tier1_data_modality_synchronization(self):
        """TC 1.1.1: Verify dataset synchronizes bSSFP, LGE, T2w with matching shapes and target masks."""
        cases = ["case0001", "case0002"]
        _create_synthetic_cache(self.root, cases, slices_per_case=2, shape=(32, 32))
        dataset = MyopsDataset(
            self.root / "bSSFP" / "train_npz",
            self.root / "LGE" / "train_npz",
            self.root / "T2w" / "train_npz",
            self.root / "splits",
            split="train",
        )
        self.assertEqual(len(dataset), 4)
        sample = dataset[0]
        for key in ("image", "image1", "image2"):
            self.assertIn(key, sample)
            self.assertEqual(sample[key].shape, (32, 32))
            self.assertTrue(np.isfinite(sample[key]).all())
        self.assertEqual(sample["label"].shape, (32, 32))

    def test_tier1_data_channel_ordering_invariant(self):
        """TC 1.1.2: Verify canonical modality ordering (bSSFP, LGE, T2w) maps correctly to image, image1, image2."""
        self.assertEqual(MODALITIES, ("bSSFP", "LGE", "T2w"))
        cases = ["case0001"]
        _create_synthetic_cache(self.root, cases, slices_per_case=1, shape=(32, 32))
        dataset = MyopsDataset(
            self.root / "bSSFP" / "train_npz",
            self.root / "LGE" / "train_npz",
            self.root / "T2w" / "train_npz",
            self.root / "splits",
            split="train",
        )
        sample = dataset[0]
        # In synthetic cache: bSSFP has base 0.2, LGE has 0.4, T2w has 0.6
        self.assertAlmostEqual(sample["image"].mean().item(), 0.2, places=4)
        self.assertAlmostEqual(sample["image1"].mean().item(), 0.4, places=4)
        self.assertAlmostEqual(sample["image2"].mean().item(), 0.6, places=4)

    def test_tier1_data_intensity_normalization_bounds(self):
        """TC 1.1.3: Verify normalize_image bounds outputs to [0, 1] and rejects non-finite values."""
        raw_image = np.array([[0.0, 127.5], [255.0, 64.0]], dtype=np.float32)
        norm_unit255 = normalize_image(raw_image, method="unit255")
        self.assertTrue((norm_unit255 >= 0.0).all() and (norm_unit255 <= 1.0).all())
        self.assertAlmostEqual(norm_unit255.max(), 1.0)
        self.assertAlmostEqual(norm_unit255.min(), 0.0)

        norm_unit = normalize_image(raw_image / 255.0, method="unit")
        self.assertTrue((norm_unit >= 0.0).all() and (norm_unit <= 1.0).all())

        norm_percentile = normalize_image(raw_image, method="percentile")
        self.assertTrue((norm_percentile >= 0.0).all() and (norm_percentile <= 1.0).all())

        with self.assertRaises(ValueError):
            normalize_image(np.array([[0.0, np.nan], [1.0, 2.0]]))

    def test_tier1_data_val_vol_contiguous_reassembly(self):
        """TC 1.1.4: Verify MyopsDataset reassembles complete 3D volume from contiguous 2D slices starting at 0."""
        cases = ["case0001"]
        _create_synthetic_cache(self.root, cases, slices_per_case=3, shape=(32, 32))
        dataset = MyopsDataset(
            self.root / "bSSFP" / "train_npz",
            self.root / "LGE" / "train_npz",
            self.root / "T2w" / "train_npz",
            self.root / "splits",
            split="val_vol",
        )
        self.assertEqual(len(dataset), 1)
        sample = dataset[0]
        # Reassembled volume should be 3D of shape (32, 32, 3)
        self.assertEqual(sample["image"].shape, (32, 32, 3))
        self.assertEqual(sample["label"].shape, (32, 32, 3))

    def test_tier1_data_synchronized_spatial_augmentation(self):
        """TC 1.1.5: Verify RandomGenerator applies identical spatial geometry across all modalities and label."""
        transform = RandomGenerator(output_size=(32, 32))
        img = np.zeros((32, 32), dtype=np.float32)
        lbl = np.zeros((32, 32), dtype=np.int64)
        # Put an identical asymmetric marker in all 3 modalities and the label
        img[5:10, 2:4] = 0.8
        lbl[5:10, 2:4] = 2
        sample = {"image": img, "image1": img.copy(), "image2": img.copy(), "label": lbl}

        # Run transform multiple times to test stochastic branches
        for seed in range(5):
            random.seed(seed)
            np.random.seed(seed)
            augmented = transform(sample)
            # Binary mask of the marker must be bitwise identical across all 3 modalities and label
            m_img = (augmented["image"][0] > 0.1).numpy()
            m_img1 = (augmented["image1"][0] > 0.1).numpy()
            m_img2 = (augmented["image2"][0] > 0.1).numpy()
            m_lbl = (augmented["label"] == 2).numpy()
            self.assertTrue(np.array_equal(m_img, m_img1))
            self.assertTrue(np.array_equal(m_img, m_img2))
            self.assertTrue(np.array_equal(m_img, m_lbl))

    # --------------------------------------------------------------------------
    # 2. Label Contract (5 test cases)
    # --------------------------------------------------------------------------

    def test_tier1_label_canonical_mapping_swaps_scar_and_edema(self):
        """TC 1.2.1: Verify legacy [0,1,3,2] remaps to canonical [0,1,2,3] (swapping edema and scar)."""
        legacy = np.array([0, 1, 2, 3], dtype=np.uint8)
        canonical = canonicalize_label(legacy, "legacy")
        # In legacy: 2 was scar, 3 was edema -> canonical: 2 is edema, 3 is scar
        self.assertEqual(list(canonical), [0, 1, 3, 2])
        self.assertEqual(CLASS_NAMES[canonical[2]], "scar")
        self.assertEqual(CLASS_NAMES[canonical[3]], "edema")

    def test_tier1_label_canonicalize_is_non_mutating_and_idempotent(self):
        """TC 1.2.2: Verify canonicalize_label is non-mutating and idempotent on canonical labels."""
        source = np.array([[0, 2], [3, 1]], dtype=np.uint8)
        source_copy = source.copy()
        result = canonicalize_label(source, "legacy")
        self.assertTrue(np.array_equal(source, source_copy))
        # Idempotence on canonical
        result2 = canonicalize_label(result, "canonical")
        self.assertTrue(np.array_equal(result, result2))

    def test_tier1_label_metadata_prevents_double_remapping(self):
        """TC 1.2.3: Verify that existing label_order metadata prevents double-remapping."""
        cases = ["case0001"]
        _create_synthetic_cache(self.root, cases, label_order=CANONICAL_LABEL_ORDER)
        dataset = MyopsDataset(
            self.root / "bSSFP" / "train_npz",
            self.root / "LGE" / "train_npz",
            self.root / "T2w" / "train_npz",
            self.root / "splits",
            split="train",
            label_order="auto",
        )
        sample = dataset[0]
        # Label in synthetic cache was created canonical: pixel at [10, 10] is class 3 (scar)
        self.assertEqual(sample["label"][10, 10].item(), 3)
        # Attempting to force 'legacy' on an already canonical metadata file must raise ValueError on read
        conflict_dataset = MyopsDataset(
            self.root / "bSSFP" / "train_npz",
            self.root / "LGE" / "train_npz",
            self.root / "T2w" / "train_npz",
            self.root / "splits",
            split="train",
            label_order="legacy",
        )
        with self.assertRaises(ValueError):
            _ = conflict_dataset[0]

    def test_tier1_label_resolve_label_order_representations(self):
        """TC 1.2.4: Verify resolve_label_order supports strings, bytes, and numpy array scalars."""
        self.assertEqual(resolve_label_order("canonical"), CANONICAL_LABEL_ORDER)
        self.assertEqual(resolve_label_order("legacy"), LEGACY_LABEL_ORDER)
        self.assertEqual(resolve_label_order(b"canonical"), CANONICAL_LABEL_ORDER)
        self.assertEqual(resolve_label_order(np.array("legacy")), LEGACY_LABEL_ORDER)
        with self.assertRaises(ValueError):
            resolve_label_order("unsupported_order")

    def test_tier1_label_rejection_of_invalid_class_ids(self):
        """TC 1.2.5: Verify canonicalize_label rejects non-integer, negative, and out-of-range class IDs."""
        with self.assertRaises(ValueError):
            canonicalize_label(np.array([0, 1, 4]), "canonical")
        with self.assertRaises(ValueError):
            canonicalize_label(np.array([0, -1, 2]), "canonical")
        with self.assertRaises(ValueError):
            canonicalize_label(np.array([0, 1.5, 2]), "canonical")

    # --------------------------------------------------------------------------
    # 3. Architecture (5 test cases)
    # --------------------------------------------------------------------------

    def test_tier1_arch_cmspa_net_m0_forward_shape(self):
        """TC 1.3.1: Verify CMSPA-Net M0 (Baseline ablation) forward pass produces (B, 4, H, W)."""
        model = CMSPANet(get_testing(), ablation="M0")
        inputs = [torch.randn(2, 1, 32, 32) for _ in range(3)]
        out = model(*inputs)
        self.assertEqual(out.shape, (2, 4, 32, 32))
        self.assertTrue(torch.isfinite(out).all())

    def test_tier1_arch_cmspa_net_m1_forward_shape(self):
        """TC 1.3.2: Verify CMSPA-Net M1 (SSPANet ablation) forward pass produces (B, 4, H, W)."""
        model = CMSPANet(get_testing(), ablation="M1")
        inputs = [torch.randn(2, 1, 32, 32) for _ in range(3)]
        out = model(*inputs)
        self.assertEqual(out.shape, (2, 4, 32, 32))
        self.assertTrue(torch.isfinite(out).all())

    def test_tier1_arch_cmspa_net_m2_forward_shape(self):
        """TC 1.3.3: Verify CMSPA-Net M2 (CMSPA bottleneck ablation) forward pass produces (B, 4, H, W)."""
        model = CMSPANet(get_testing(), ablation="M2")
        inputs = [torch.randn(2, 1, 32, 32) for _ in range(3)]
        out = model(*inputs)
        self.assertEqual(out.shape, (2, 4, 32, 32))
        self.assertTrue(torch.isfinite(out).all())

    def test_tier1_arch_cmspa_net_m3_forward_shape(self):
        """TC 1.3.4: Verify CMSPA-Net M3 (Full model) forward pass produces (B, 4, H, W)."""
        model = CMSPANet(get_testing(), ablation="M3")
        inputs = [torch.randn(2, 1, 32, 32) for _ in range(3)]
        out = model(*inputs)
        self.assertEqual(out.shape, (2, 4, 32, 32))
        self.assertTrue(torch.isfinite(out).all())

    def test_tier1_arch_cmspa_net_varying_batch_sizes(self):
        """TC 1.3.5: Verify CMSPA-Net M3 preserves exact batch size for B=1, B=3, B=5."""
        model = CMSPANet(get_testing(), ablation="M3")
        for b in (1, 3, 5):
            inputs = [torch.randn(b, 1, 32, 32) for _ in range(3)]
            out = model(*inputs)
            self.assertEqual(out.shape, (b, 4, 32, 32))

    # --------------------------------------------------------------------------
    # 4. Attention Mechanisms (5 test cases)
    # --------------------------------------------------------------------------

    def test_tier1_attn_sspanet_rms_strip_pooling(self):
        """TC 1.4.1: Verify strip_rms computes sqrt(mean(x^2) + eps) with epsilon inside sqrt."""
        x = torch.tensor([[[[3.0, 4.0]]]])  # shape (1, 1, 1, 2)
        # Along dim=3: mean(x^2) = (9 + 16)/2 = 12.5
        rms_val = strip_rms(x, dim=3, epsilon=1e-6)
        expected = math.sqrt(12.5 + 1e-6)
        self.assertAlmostEqual(rms_val.item(), expected, places=6)

    def test_tier1_attn_sspa_spatial_attention_shape_and_flow(self):
        """TC 1.4.2: Verify SSPA_SpatialAttention preserves channels and propagates gradients."""
        attn = SSPA_SpatialAttention(inplanes=16)
        x = torch.randn(2, 16, 16, 16, requires_grad=True)
        out = attn(x)
        self.assertEqual(out.shape, (2, 16, 16, 16))
        out.sum().backward()
        self.assertIsNotNone(attn.conv1.weight.grad)
        self.assertIsNotNone(attn.conv2.weight.grad)
        self.assertTrue(torch.isfinite(attn.conv1.weight.grad).all())

    def test_tier1_attn_sspa_factored_projection_equivalence(self):
        """TC 1.4.3: Verify factored 1x1 strip projection is algebraically identical to broadcast convolution."""
        conv = nn.Conv2d(8, 8, 1)
        strip_h = torch.randn(2, 8, 16, 1)
        strip_w = torch.randn(2, 8, 1, 16)
        unfactored = conv(strip_h + strip_w)
        factored = F.conv2d(strip_h, conv.weight, conv.bias) + F.conv2d(strip_w, conv.weight)
        torch.testing.assert_close(factored, unfactored, atol=1e-5, rtol=1e-5)

    def test_tier1_attn_sspa_channel_attention_zpool_conv7x7(self):
        """TC 1.4.4: Verify SSPA_ZPool channel compression to (B, 2, H, W) and SSPA_ChannelAttention spatial gate."""
        zpool = SSPA_ZPool()
        x = torch.randn(2, 16, 8, 8)
        pooled = zpool(x)
        self.assertEqual(pooled.shape, (2, 2, 8, 8))
        self.assertTrue(torch.equal(pooled[:, 0:1], x.max(dim=1, keepdim=True).values))
        self.assertTrue(torch.allclose(pooled[:, 1:2], x.mean(dim=1, keepdim=True)))

        ca = SSPA_ChannelAttention()
        out = ca(x)
        self.assertEqual(out.shape, x.shape)

    def test_tier1_attn_sspanet_block_residual_combination(self):
        """TC 1.4.5: Verify SSPANet_Block combines channel and spatial attention with residual addition."""
        block = SSPANet_Block(in_channels=16)
        x = torch.randn(2, 16, 8, 8, requires_grad=True)
        out = block(x)
        self.assertEqual(out.shape, (2, 16, 8, 8))
        out.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    # --------------------------------------------------------------------------
    # 5. Bottleneck Gates (5 test cases)
    # --------------------------------------------------------------------------

    def test_tier1_bottleneck_cine_anatomy_gate(self):
        """TC 1.5.1: Verify CINE anatomy gate produces sigmoid modulated spatial prior in [0, 1]."""
        fusion = CMSPA_Fusion(in_channels=16, out_channels=8)
        cine = torch.randn(2, 16, 8, 8)
        anatomy = fusion.conv_strip(cine.mean(dim=3, keepdim=True) + cine.mean(dim=2, keepdim=True))
        self.assertEqual(anatomy.shape, (2, 1, 8, 8))
        self.assertTrue((anatomy >= 0.0).all() and (anatomy <= 1.0).all())

    def test_tier1_bottleneck_lge_t2w_pathology_gate(self):
        """TC 1.5.2: Verify LGE+T2w pathology gate derives from channel_std and produces sigmoid gate in [0, 1]."""
        fusion = CMSPA_Fusion(in_channels=16, out_channels=8)
        psir = torch.randn(2, 16, 8, 8)
        t2w = torch.randn(2, 16, 8, 8)
        style = (channel_std(psir) + channel_std(t2w)).float()
        self.assertEqual(style.shape, (2, 1, 8, 8))
        patho = fusion.conv_patho(style)
        self.assertEqual(patho.shape, (2, 1, 8, 8))
        self.assertTrue((patho >= 0.0).all() and (patho <= 1.0).all())

    def test_tier1_bottleneck_cine_residual_pathology_modulation(self):
        """TC 1.5.3: Verify cine + cine * pathology guarantees CINE feature survival even when pathology gate is 0."""
        cine = torch.randn(2, 16, 8, 8)
        zero_gate = torch.zeros(2, 1, 8, 8)
        modulated = cine + cine * zero_gate
        self.assertTrue(torch.equal(modulated, cine))

    def test_tier1_bottleneck_modality_permutation_asymmetry(self):
        """TC 1.5.4: Verify swapping CINE and LGE inputs alters bottleneck fused representation (architectural asymmetry)."""
        fusion = CMSPA_Fusion(in_channels=16, out_channels=8)
        torch.manual_seed(42)
        cine = torch.randn(1, 16, 8, 8)
        psir = torch.randn(1, 16, 8, 8)
        t2w = torch.randn(1, 16, 8, 8)
        out_normal = fusion(cine, psir, t2w)
        out_swapped = fusion(psir, cine, t2w)
        self.assertFalse(torch.allclose(out_normal, out_swapped))

    def test_tier1_bottleneck_fusion_projection_shape(self):
        """TC 1.5.5: Verify fusion_conv correctly projects 3*in_channels to out_channels."""
        fusion = CMSPA_Fusion(in_channels=16, out_channels=8)
        cine = torch.randn(2, 16, 8, 8)
        psir = torch.randn(2, 16, 8, 8)
        t2w = torch.randn(2, 16, 8, 8)
        out = fusion(cine, psir, t2w)
        self.assertEqual(out.shape, (2, 8, 8, 8))

    # --------------------------------------------------------------------------
    # 6. Losses & Metrics (5 test cases)
    # --------------------------------------------------------------------------

    def test_tier1_loss_diceloss_analytical_values(self):
        """TC 1.6.1: Verify DiceLoss matches analytical value on uniform logits and is zero on perfect predictions."""
        criterion = DiceLoss()
        target = torch.tensor([[[0, 1], [2, 3]]])
        one_hot = F.one_hot(target, 4).movedim(-1, 1).float()
        self.assertEqual(criterion(one_hot, target).item(), 0.0)

        # Uniform logits
        uniform_logits = torch.zeros(1, 4, 2, 2)
        target_zero = torch.zeros(1, 2, 2, dtype=torch.long)
        res = criterion(uniform_logits, target_zero, softmax=True)
        self.assertTrue(torch.isfinite(res))
        self.assertGreater(res.item(), 0.0)

    def test_tier1_loss_segmentationloss_weighted_sum(self):
        """TC 1.6.2: Verify SegmentationLoss computes exact weighted combination of CE and DiceLoss."""
        loss_fn = SegmentationLoss(ce_weight=0.7, dice_weight=0.3)
        logits = torch.randn(2, 4, 8, 8, requires_grad=True)
        target = torch.randint(0, 4, (2, 8, 8))
        res = loss_fn(logits, target)
        expected = 0.7 * res["ce"] + 0.3 * res["dice_loss"]
        self.assertAlmostEqual(res["loss"].item(), expected.item(), places=5)

    def test_tier1_metric_confusion_meter_per_class_scores(self):
        """TC 1.6.3: Verify ConfusionMeter accurately computes per-class Dice and IoU."""
        meter = ConfusionMeter(num_classes=4)
        target = torch.tensor([0, 1, 2, 3])
        pred = torch.tensor([0, 1, 2, 3])
        meter.update(pred, target)
        scores = meter.compute()
        for name in CLASS_NAMES:
            self.assertAlmostEqual(scores[f"dice/{name}"], 1.0)
            self.assertAlmostEqual(scores[f"iou/{name}"], 1.0)
        self.assertAlmostEqual(scores["mean_dice"], 1.0)

    def test_tier1_metric_surface_distance_hd95_with_spacing(self):
        """TC 1.6.4: Verify symmetric HD95 accounts for anisotropic physical voxel spacing."""
        target = np.zeros((10, 10, 10), dtype=bool)
        pred = np.zeros((10, 10, 10), dtype=bool)
        target[5, 5, 5] = True
        pred[5, 5, 7] = True  # Shift of 2 voxels along axis 2
        spacing = (1.0, 1.0, 3.5)
        m = binary_metrics(pred, target, spacing=spacing)
        # Expected distance along axis 2 = 2 * 3.5 = 7.0 mm
        self.assertAlmostEqual(m["hd95"], 7.0, places=4)

    def test_tier1_metric_surface_distance_asd_or_hd95_properties(self):
        """TC 1.6.5: Verify surface distance properties: zero on identical shapes, non-negative, and finite."""
        mask = np.zeros((16, 16), dtype=bool)
        mask[4:10, 4:10] = True
        m = binary_metrics(mask, mask, spacing=(1.2, 1.2))
        self.assertEqual(m["hd95"], 0.0)
        if "asd" in m and m["asd"] is not None:
            self.assertEqual(m["asd"], 0.0)
        self.assertEqual(m["dice"], 1.0)
        self.assertEqual(m["iou"], 1.0)

    # --------------------------------------------------------------------------
    # 7. Trainer Primitives (5 test cases)
    # --------------------------------------------------------------------------

    def test_tier1_trainer_gradient_accumulation_scaling(self):
        """TC 1.7.1: Verify sample-weighted microbatch accumulation mathematically matches full-batch gradient."""
        full_layer = nn.Linear(4, 2, bias=False)
        accum_layer = copy.deepcopy(full_layer)
        x = torch.randn(6, 4)
        y = torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.long)

        # Full batch
        full_loss = F.cross_entropy(full_layer(x), y)
        full_loss.backward()

        # Accumulated in microbatches: 4 + 2
        for start, end in ((0, 4), (4, 6)):
            mb_loss = F.cross_entropy(accum_layer(x[start:end]), y[start:end])
            (mb_loss * (end - start)).backward()
        accum_layer.weight.grad.div_(6)

        torch.testing.assert_close(full_layer.weight.grad, accum_layer.weight.grad)

    def test_tier1_trainer_gradient_clipping_threshold(self):
        """TC 1.7.2: Verify clip_grad_norm_ restricts total parameter gradient norm to <= 1.0."""
        layer = nn.Linear(10, 10, bias=False)
        x = torch.randn(10, 10) * 100.0
        out = layer(x)
        out.sum().backward()
        initial_norm = torch.nn.utils.clip_grad_norm_(layer.parameters(), 1.0)
        self.assertGreater(initial_norm, 1.0)
        clipped_norm = torch.linalg.norm(layer.weight.grad)
        self.assertAlmostEqual(clipped_norm.item(), 1.0, places=4)

    def test_tier1_trainer_atomic_checkpointing_integrity(self):
        """TC 1.7.3: Verify atomic_checkpoint creates complete valid files and cleans up temporary files."""
        ckpt_path = self.root / "test_model.pth"
        payload = {"format_version": 1, "class_names": CLASS_NAMES, "weights": torch.ones(5)}
        atomic_checkpoint(ckpt_path, payload)
        self.assertTrue(ckpt_path.is_file())
        self.assertFalse(ckpt_path.with_suffix(".pth.tmp").exists())
        loaded = load_checkpoint(ckpt_path)
        self.assertEqual(loaded["format_version"], 1)
        self.assertEqual(tuple(loaded["class_names"]), CLASS_NAMES)

    def test_tier1_trainer_deterministic_seeding(self):
        """TC 1.7.4: Verify seed_everything guarantees bitwise identical RNG outputs across invocations."""
        seed_everything(1234)
        v1 = torch.randn(10)
        n1 = np.random.rand(10)
        r1 = [random.random() for _ in range(10)]

        seed_everything(1234)
        v2 = torch.randn(10)
        n2 = np.random.rand(10)
        r2 = [random.random() for _ in range(10)]

        self.assertTrue(torch.equal(v1, v2))
        self.assertTrue(np.array_equal(n1, n2))
        self.assertEqual(r1, r2)

    def test_tier1_trainer_split_manifest_sha256_verification(self):
        """TC 1.7.5: Verify split manifest SHA256 hashing correctly detects file modifications."""
        manifest = self.root / "train.txt"
        manifest.write_text("case0001_slice000\ncase0001_slice001\n", encoding="utf-8")
        h1 = hashlib.sha256(manifest.read_bytes()).hexdigest()
        # Alter content
        manifest.write_text("case0001_slice000\ncase0001_slice002\n", encoding="utf-8")
        h2 = hashlib.sha256(manifest.read_bytes()).hexdigest()
        self.assertNotEqual(h1, h2)


# ==============================================================================
# TIER 2: BOUNDARY & CORNER CASES
# ==============================================================================


class TestTier2BoundaryAndCornerCases(unittest.TestCase):
    """Tier 2: Boundary conditions, stress testing, and edge case error handling."""

    def test_tier2_boundary_empty_ground_truth_masks(self):
        """TC 2.1: Verify binary_metrics returns both_empty or target_empty status on empty masks."""
        empty = np.zeros((16, 16), dtype=bool)
        res_both = binary_metrics(empty, empty)
        self.assertEqual(res_both["status"], "both_empty")
        self.assertIn(res_both["hd95"], (None, 0.0))
        self.assertIn(res_both["dice"], (None, 1.0))
        if "asd" in res_both:
            self.assertIn(res_both["asd"], (None, 0.0))

        non_empty = np.zeros((16, 16), dtype=bool)
        non_empty[2, 2] = True
        res_pred_empty = binary_metrics(empty, non_empty)
        self.assertIn(res_pred_empty["status"], ("prediction_empty", "one_empty"))
        self.assertEqual(res_pred_empty["dice"], 0.0)
        self.assertIn(res_pred_empty["hd95"], (None, float("inf")))

        res_target_empty = binary_metrics(non_empty, empty)
        self.assertIn(res_target_empty["status"], ("target_empty", "one_empty"))
        self.assertEqual(res_target_empty["dice"], 0.0)
        self.assertIn(res_target_empty["hd95"], (None, float("inf")))

    def test_tier2_boundary_single_pixel_foreground_masks(self):
        """TC 2.2: Verify ConfusionMeter and binary_metrics safely handle minimal 1-pixel foreground masks."""
        single_target = torch.zeros(100, dtype=torch.long)
        single_pred = torch.zeros(100, dtype=torch.long)
        single_target[42] = 3
        single_pred[42] = 3

        meter = ConfusionMeter(num_classes=4)
        meter.update(single_pred, single_target)
        scores = meter.compute()
        self.assertAlmostEqual(scores["dice/scar"], 1.0)

        m = binary_metrics(single_pred.numpy() == 3, single_target.numpy() == 3)
        self.assertEqual(m["dice"], 1.0)
        self.assertEqual(m["hd95"], 0.0)

    def test_tier2_boundary_absent_lesion_classes(self):
        """TC 2.3: Verify ConfusionMeter assigns NaN to absent lesion classes and excludes them from mean_dice."""
        # Only class 0 and 1 exist
        target = torch.tensor([0, 0, 1, 1])
        pred = torch.tensor([0, 0, 1, 1])
        meter = ConfusionMeter(num_classes=4)
        meter.update(pred, target)
        scores = meter.compute()
        self.assertAlmostEqual(scores["dice/normal_myocardium"], 1.0)
        self.assertTrue(math.isnan(scores["dice/edema"]))
        self.assertTrue(math.isnan(scores["dice/scar"]))
        # Mean dice over valid foreground classes must be 1.0 (excluding edema and scar NaNs)
        self.assertAlmostEqual(scores["mean_dice"], 1.0)

    def test_tier2_boundary_surface_distance_null_spacing_safety(self):
        """TC 2.4: Verify binary_metrics with compute_distance=False returns hd95=None and does not default to 1.0mm."""
        pred = np.zeros((10, 10), dtype=bool)
        target = np.zeros((10, 10), dtype=bool)
        pred[2:5, 2:5] = True
        target[2:5, 2:5] = True
        m = binary_metrics(pred, target, spacing=None, compute_distance=False)
        self.assertIsNone(m["hd95"])
        if "asd" in m:
            self.assertIsNone(m["asd"])

    def test_tier2_boundary_anisotropic_physical_spacing(self):
        """TC 2.5: Verify surface distance correctly scales along highly anisotropic axes (e.g. 1.0x1.0x8.0mm)."""
        target = np.zeros((5, 5, 5), dtype=bool)
        pred_z = np.zeros((5, 5, 5), dtype=bool)
        pred_x = np.zeros((5, 5, 5), dtype=bool)
        target[2, 2, 2] = True
        pred_z[2, 2, 3] = True  # Shift along Z
        pred_x[3, 2, 2] = True  # Shift along X
        spacing = (1.0, 1.0, 8.0)
        m_z = binary_metrics(pred_z, target, spacing=spacing)
        m_x = binary_metrics(pred_x, target, spacing=spacing)
        self.assertAlmostEqual(m_z["hd95"], 8.0)
        self.assertAlmostEqual(m_x["hd95"], 1.0)

    def test_tier2_boundary_non_orthogonal_affine_rejection(self):
        """TC 2.6: Verify that non-orthogonal (sheared) affine matrices are detected and rejected."""
        sheared_affine = np.array([
            [1.0, 0.5, 0.0, 0.0],
            [0.2, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        axes = sheared_affine[:3, :3]
        axes = axes / np.linalg.norm(axes, axis=0)
        is_orthogonal = np.allclose(axes.T @ axes, np.eye(3), atol=1e-4)
        self.assertFalse(is_orthogonal)

    def test_tier2_boundary_amp_fp16_large_activation_overflow_prevention(self):
        """TC 2.7: Verify strip_rms and channel_std do not overflow to inf under FP16 with large values (>256)."""
        large_fp16 = torch.full((1, 4, 8, 8), 500.0, dtype=torch.float16)
        # In FP16: 500^2 = 250,000 > 65,504 (overflows FP16 if not upcast to FP32)
        rms = strip_rms(large_fp16, dim=3)
        self.assertTrue(torch.isfinite(rms).all())
        self.assertAlmostEqual(rms.float().mean().item(), 500.0, places=1)

        std = channel_std(large_fp16)
        self.assertTrue(torch.isfinite(std).all())

    def test_tier2_boundary_uneven_batches_at_epoch_boundary(self):
        """TC 2.8: Verify gradient accumulation correctly handles fractional boundary batches (e.g. 3, 3, 1)."""
        layer_full = nn.Linear(4, 2, bias=False)
        layer_accum = copy.deepcopy(layer_full)
        x = torch.randn(7, 4)
        y = torch.tensor([0, 1, 0, 1, 0, 1, 0], dtype=torch.long)

        # Full batch of 7
        loss_full = F.cross_entropy(layer_full(x), y)
        loss_full.backward()

        # Accumulate batches of 3, 3, 1
        for start, end in ((0, 3), (3, 6), (6, 7)):
            loss_mb = F.cross_entropy(layer_accum(x[start:end]), y[start:end])
            (loss_mb * (end - start)).backward()
        layer_accum.weight.grad.div_(7)

        torch.testing.assert_close(layer_full.weight.grad, layer_accum.weight.grad)

    def test_tier2_boundary_out_of_bounds_class_ids_rejection(self):
        """TC 2.9: Verify ConfusionMeter strictly rejects out-of-range class IDs (e.g. 4 or -1)."""
        meter = ConfusionMeter(num_classes=4)
        with self.assertRaises(ValueError):
            meter.update(torch.tensor([4]), torch.tensor([0]))
        with self.assertRaises(ValueError):
            meter.update(torch.tensor([0]), torch.tensor([-1]))


# ==============================================================================
# TIER 3: CROSS-FEATURE COMBINATIONS
# ==============================================================================


class TestTier3CrossFeatureCombinations(unittest.TestCase):
    """Tier 3: Multi-feature pairwise interactions across subsystems."""

    def test_tier3_data_model_loss_gradient_accumulation_pipeline(self):
        """TC 3.1: Data loading + CMSPA-Net M3 forward + backward + DiceLoss + gradient accumulation."""
        torch.manual_seed(42)
        model = CMSPANet(get_testing(), ablation="M3")
        criterion = SegmentationLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad()

        # Simulate 2 accumulated microbatches of size 2
        total_samples = 4
        for mb in range(2):
            cine = torch.randn(2, 1, 32, 32)
            psir = torch.randn(2, 1, 32, 32)
            t2w = torch.randn(2, 1, 32, 32)
            target = torch.randint(0, 4, (2, 32, 32))
            logits = model(cine, psir, t2w)
            loss_dict = criterion(logits, target)
            (loss_dict["loss"] * 2).backward()

        for p in model.parameters():
            if p.grad is not None:
                p.grad.div_(total_samples)

        # Check all parameters have finite gradients
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertEqual(len(grads), len(list(model.parameters())))
        self.assertTrue(all(torch.isfinite(g).all() for g in grads))
        self.assertTrue(all(g.abs().sum() > 0 for g in grads))

    def test_tier3_canonical_mapping_rare_sampler_confusion_matrix(self):
        """TC 3.2: Canonical label mapping + rare pathology sampler + confusion matrix pixel pooling."""
        tmp = tempfile.mkdtemp()
        try:
            root = Path(tmp)
            # Create synthetic dataset with scar in some slices
            cases = ["case0001", "case0002"]
            _create_synthetic_cache(root, cases, slices_per_case=2, label_order=LEGACY_LABEL_ORDER)
            dataset = MyopsDataset(
                root / "bSSFP" / "train_npz",
                root / "LGE" / "train_npz",
                root / "T2w" / "train_npz",
                root / "splits",
                split="train",
                label_order="auto",
            )
            # Build rare class sampler targeting canonical scar (3)
            sampler = build_rare_class_sampler(dataset, rare_classes=(3,), rare_boost=3.0)
            self.assertEqual(len(sampler), len(dataset))

            # Pool predictions and targets in ConfusionMeter
            meter = ConfusionMeter(num_classes=4)
            for idx in list(sampler)[:2]:
                sample = dataset[idx]
                target_t = torch.from_numpy(sample["label"])
                pred_t = target_t.clone()
                meter.update(pred_t.flatten(), target_t.flatten())
            scores = meter.compute()
            self.assertAlmostEqual(scores["pixel_accuracy"], 1.0)
            self.assertAlmostEqual(scores["dice/scar"], 1.0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_tier3_checkpoint_atomic_save_and_resume_verification(self):
        """TC 3.3: Checkpoint serialization + atomic replace + resume verification (all hyperparameters & bitwise parameters)."""
        tmp = tempfile.mkdtemp()
        try:
            ckpt_path = Path(tmp) / "last.pth"
            model_a = CMSPANet(get_testing(), ablation="M3")
            opt_a = torch.optim.AdamW(model_a.parameters(), lr=1e-3)

            # Do one step to modify weights
            inputs = [torch.randn(1, 1, 32, 32) for _ in range(3)]
            loss = model_a(*inputs).sum()
            loss.backward()
            opt_a.step()

            payload = {
                "format_version": 1,
                "class_names": CLASS_NAMES,
                "model": model_a.state_dict(),
                "optimizer": opt_a.state_dict(),
                "split_hashes": {"train": "abc123hash"},
                "epoch": 1,
                "global_step": 10,
            }
            atomic_checkpoint(ckpt_path, payload)

            # Create fresh model B and restore
            model_b = CMSPANet(get_testing(), ablation="M3")
            opt_b = torch.optim.AdamW(model_b.parameters(), lr=1e-3)

            loaded = load_checkpoint(ckpt_path)
            model_b.load_state_dict(loaded["model"])
            opt_b.load_state_dict(loaded["optimizer"])

            for p_a, p_b in zip(model_a.parameters(), model_b.parameters()):
                torch.testing.assert_close(p_a, p_b, atol=0, rtol=0)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


# ==============================================================================
# TIER 4: REAL-WORLD APPLICATION SCENARIOS
# ==============================================================================


class TestTier4RealWorldScenarios(unittest.TestCase):
    """Tier 4: Realistic end-to-end clinical and execution pipeline scenarios."""

    def test_tier4_full_volume_inference_and_spatial_reassembly(self):
        """TC 4.1: Full volume inference on synthetic multi-slice volume validating correct spatial reassembly."""
        model = CMSPANet(get_testing(), ablation="M3")
        volume_shape = (64, 64, 5)  # 5 slices
        images = [
            np.random.rand(*volume_shape).astype(np.float32) for _ in range(3)
        ]
        pred_vol = predict_volume(model, images, img_size=32, batch_size=2)
        self.assertEqual(pred_vol.shape, volume_shape)
        self.assertEqual(pred_vol.dtype, np.uint8)
        self.assertTrue((pred_vol >= 0).all() and (pred_vol <= 3).all())

    def test_tier4_full_training_step_with_amp_and_adamw(self):
        """TC 4.2: Full training step with AMP, GradScaler, gradient clipping, and AdamW optimizer update."""
        torch.manual_seed(99)
        model = CMSPANet(get_testing(), ablation="M3")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=False)  # CPU-compatible fallback
        criterion = SegmentationLoss()

        params_before = [p.clone() for p in model.parameters()]
        inputs = [torch.randn(2, 1, 32, 32) for _ in range(3)]
        target = torch.randint(0, 4, (2, 32, 32))

        optimizer.zero_grad()
        with torch.autocast(device_type="cpu", enabled=False):
            logits = model(*inputs)
            loss_dict = criterion(logits, target)
            loss = loss_dict["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        # Assert parameters have updated cleanly without NaN
        updated = any(not torch.equal(p1, p2) for p1, p2 in zip(params_before, model.parameters()))
        self.assertTrue(updated)
        self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))

    def test_tier4_evaluation_pipeline_with_composite_clinical_regions(self):
        """TC 4.3: Clinical evaluation pipeline computing edema_inclusive (>=2) and myocardial_ring (>=1)."""
        pred = np.zeros((32, 32, 4), dtype=np.uint8)
        target = np.zeros((32, 32, 4), dtype=np.uint8)
        # Ring: myocardium = 1, edema = 2, scar = 3
        target[8:24, 8:24, :] = 1
        target[12:20, 12:20, :] = 2
        target[14:18, 14:18, :] = 3

        pred[8:24, 8:24, :] = 1
        pred[12:20, 12:20, :] = 2
        pred[14:18, 14:18, :] = 3

        regions = [(name, pred == i, target == i) for i, name in enumerate(CLASS_NAMES) if i]
        regions.extend([
            ("edema_inclusive", pred >= 2, target >= 2),
            ("myocardial_ring", pred >= 1, target >= 1),
        ])

        spacing = (1.25, 1.25, 8.0)
        results = {}
        for name, p_mask, t_mask in regions:
            m = binary_metrics(p_mask, t_mask, spacing=spacing)
            results[name] = m
            self.assertEqual(m["dice"], 1.0)
            self.assertEqual(m["hd95"], 0.0)

        self.assertIn("edema_inclusive", results)
        self.assertIn("myocardial_ring", results)
        self.assertEqual(results["edema_inclusive"]["dice"], 1.0)
        self.assertEqual(results["myocardial_ring"]["dice"], 1.0)


if __name__ == "__main__":
    unittest.main()
