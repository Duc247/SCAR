"""Numerical and inference contracts, independent of the expensive backbone."""
import copy
import math
import unittest

import numpy as np
import pytest
import torch
from torch import nn

from training.loss import DiceLoss, SegmentationLoss, build_loss
from training.metrics import ConfusionMeter, binary_metrics, calculate_metric_percase


class LossTests(unittest.TestCase):
    def test_uniform_logits_have_analytic_ce_and_dice(self):
        logits = torch.zeros(1, 4, 2, 3, requires_grad=True)
        target = torch.zeros(1, 2, 3, dtype=torch.long)
        result = SegmentationLoss()(logits, target)
        smooth = 1e-5
        background_score = (3 + smooth) / (6.375 + smooth)
        absent_score = smooth / (0.375 + smooth)
        expected_dice = 1 - (background_score + 3 * absent_score) / 4
        self.assertAlmostEqual(result['ce'].item(), math.log(4), places=6)
        self.assertAlmostEqual(result['dice_loss'].item(), expected_dice, places=6)
        self.assertAlmostEqual(result['loss'].item(),
                               0.5 * (math.log(4) + expected_dice), places=6)
        result['loss'].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertTrue((logits.grad.abs().sum((0, 2, 3)) > 0).all())

    def test_perfect_probabilities_have_zero_dice_loss(self):
        target = torch.tensor([[[0, 1], [2, 3]]])
        probabilities = torch.nn.functional.one_hot(target, 4).movedim(-1, 1).float()
        self.assertEqual(DiceLoss()(probabilities, target).item(), 0)

    def test_sample_weighted_uneven_accumulation_matches_full_batch(self):
        torch.manual_seed(913)
        full = nn.Conv2d(2, 4, kernel_size=1)
        accumulated = copy.deepcopy(full)
        inputs = torch.randn(5, 2, 5, 7)
        target = torch.randint(0, 4, (5, 5, 7))
        criterion = SegmentationLoss()
        whole = criterion(full(inputs), target)
        whole['loss'].backward()
        weighted_components = {key: 0.0 for key in whole}
        for start, stop in ((0, 2), (2, 4), (4, 5)):
            result = criterion(accumulated(inputs[start:stop]), target[start:stop])
            fraction = (stop - start) / len(inputs)
            (result['loss'] * fraction).backward()
            for key in result:
                weighted_components[key] += result[key].detach().item() * fraction
        for key in whole:
            self.assertAlmostEqual(weighted_components[key], whole[key].item(), places=6)
        for full_parameter, accumulated_parameter in zip(full.parameters(), accumulated.parameters()):
            torch.testing.assert_close(full_parameter.grad, accumulated_parameter.grad,
                                       rtol=1e-5, atol=1e-7)

    def test_invalid_dice_weights_fail_clearly(self):
        target = torch.zeros(1, 2, 2, dtype=torch.long)
        probabilities = torch.full((1, 4, 2, 2), 0.25)
        for weights in ([0, 0, 0, 0], [1, -1, 1, 1], [1, 1], [1, 1, 1, float('nan')]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                DiceLoss()(probabilities, target, weight=weights)

    def test_loss_factory_and_invalid_weights(self):
        self.assertIsInstance(build_loss('ce_dice', num_classes=4), SegmentationLoss)
        for weights in ((-1, 2), (0, 0), (float('nan'), 1), (1, float('inf'))):
            with self.assertRaises(ValueError):
                SegmentationLoss(ce_weight=weights[0], dice_weight=weights[1])
        with self.assertRaises(ValueError):
            build_loss('unknown')


class MetricTests(unittest.TestCase):
    def test_invalid_class_ids_cannot_alias_valid_confusion_entries(self):
        meter = ConfusionMeter()
        with self.assertRaises(ValueError):
            meter.update(torch.tensor([4]), torch.tensor([0]))
        with self.assertRaises(ValueError):
            meter.update(torch.tensor([0]), torch.tensor([-1]))
        with self.assertRaises(TypeError):
            meter.update(torch.tensor([0.5]), torch.tensor([0]))
        with self.assertRaises(ValueError):
            meter.update(torch.tensor([[0]]), torch.tensor([0]))
        self.assertEqual(meter.matrix.sum().item(), 0)

    def test_confusion_metrics_pool_pixels_and_exclude_absent_class(self):
        meter = ConfusionMeter()
        target = torch.tensor([0, 1, 1, 2])
        prediction = torch.tensor([0, 1, 2, 2])
        meter.update(prediction[:1], target[:1])
        meter.update(prediction[1:], target[1:])
        scores = meter.compute()
        self.assertAlmostEqual(scores['dice/background'], 1)
        self.assertAlmostEqual(scores['dice/normal_myocardium'], 2 / 3)
        self.assertAlmostEqual(scores['dice/edema'], 2 / 3)
        self.assertTrue(math.isnan(scores['dice/scar']))
        self.assertAlmostEqual(scores['mean_dice'], 2 / 3)
        self.assertAlmostEqual(scores['mean_iou'], 0.5)
        self.assertAlmostEqual(scores['pixel_accuracy'], 0.75)

    def test_both_empty_are_undefined_not_a_perfect_lesion(self):
        empty = np.zeros((4, 5), dtype=bool)
        result = binary_metrics(empty, empty, empty_mode="undefined")
        self.assertEqual(result, {'dice': None, 'iou': None, 'hd95': None, 'asd': None,
                                  'status': 'both_empty'})
        dice, hd95 = calculate_metric_percase(empty, empty)
        self.assertTrue(np.isnan(dice))
        self.assertTrue(np.isnan(hd95))

    def test_small_false_positive_and_missed_lesion_are_not_perfect(self):
        empty = np.zeros((4, 5), dtype=bool)
        lesion = empty.copy()
        lesion[2, 3] = True
        for prediction, target, status in ((lesion, empty, 'target_empty'),
                                           (empty, lesion, 'prediction_empty')):
            with self.subTest(status=status):
                result = binary_metrics(prediction, target, empty_mode="undefined")
                self.assertEqual(result['dice'], 0)
                self.assertEqual(result['iou'], 0)
                self.assertIsNone(result['hd95'])
                self.assertIsNone(result['asd'])
                self.assertEqual(result['status'], status)
                self.assertEqual(calculate_metric_percase(prediction, target), (0, np.inf))

    def test_identical_nonempty_masks_have_perfect_overlap_and_zero_distance(self):
        mask = np.zeros((7, 9), dtype=bool)
        mask[2:5, 3:6] = True
        self.assertEqual(binary_metrics(mask, mask),
                         {'dice': 1, 'iou': 1, 'hd95': 0, 'asd': 0.0, 'status': 'ok'})

    def test_empty_mask_behaviors_and_asd(self):
        empty = np.zeros((4, 5), dtype=bool)
        lesion = empty.copy()
        lesion[2, 3] = True

        # Both empty with empty_mode="defined" -> dice=1.0, iou=1.0, hd95=0.0, asd=0.0, status="both_empty"
        both_empty = binary_metrics(empty, empty, empty_mode="defined")
        self.assertEqual(both_empty, {
            "dice": 1.0,
            "iou": 1.0,
            "hd95": 0.0,
            "asd": 0.0,
            "status": "both_empty",
        })

        # One empty with empty_mode="defined" -> dice=0.0, iou=0.0, hd95=inf, asd=inf, status="one_empty"
        for pred, true in ((lesion, empty), (empty, lesion)):
            res = binary_metrics(pred, true, empty_mode="defined")
            self.assertEqual(res["dice"], 0.0)
            self.assertEqual(res["iou"], 0.0)
            self.assertEqual(res["hd95"], float("inf"))
            self.assertEqual(res["asd"], float("inf"))
            self.assertEqual(res["status"], "one_empty")

        # Default empty_mode="undefined" -> hd95 is None, asd is None, status="both_empty"
        both_default = binary_metrics(empty, empty)
        self.assertIsNone(both_default["hd95"])
        self.assertIsNone(both_default["asd"])
        self.assertEqual(both_default["status"], "both_empty")

        # compute_distance=False -> hd95 is None, asd is None, status="missing_physical_geometry"
        res_no_dist = binary_metrics(lesion, lesion, compute_distance=False)
        self.assertIsNone(res_no_dist["hd95"])
        self.assertIsNone(res_no_dist["asd"])
        self.assertEqual(res_no_dist["status"], "missing_physical_geometry")

        # compute_distance=False with spacing -> hd95 is None, asd is None, status="ok"
        res_spacing_no_dist = binary_metrics(lesion, lesion, spacing=(1.0, 1.0), compute_distance=False)
        self.assertIsNone(res_spacing_no_dist["hd95"])
        self.assertIsNone(res_spacing_no_dist["asd"])
        self.assertEqual(res_spacing_no_dist["status"], "ok")

    def test_asd_identical_shapes_is_zero(self):
        mask2d = np.zeros((10, 10), dtype=bool)
        mask2d[3:7, 4:8] = True
        res2d = binary_metrics(mask2d, mask2d)
        self.assertEqual(res2d["asd"], 0.0)
        self.assertEqual(res2d["hd95"], 0.0)

        mask3d = np.zeros((8, 8, 8), dtype=bool)
        mask3d[2:6, 2:6, 2:6] = True
        res3d = binary_metrics(mask3d, mask3d, spacing=(1.5, 1.5, 2.0))
        self.assertEqual(res3d["asd"], 0.0)
        self.assertEqual(res3d["hd95"], 0.0)

    def test_asd_translated_shapes_known_analytical_distance(self):
        # 2D translation: 1x3 bar translated by 3 voxels along axis 0
        target2d = np.zeros((12, 12), dtype=bool)
        target2d[4, 5:8] = True
        pred2d = np.zeros((12, 12), dtype=bool)
        pred2d[7, 5:8] = True
        res2d = binary_metrics(pred2d, target2d)
        self.assertAlmostEqual(res2d["asd"], 3.0, places=6)
        self.assertAlmostEqual(res2d["hd95"], 3.0, places=6)

        # 3D translation: single voxel translated by 2 voxels along axis 2
        target3d = np.zeros((6, 6, 6), dtype=bool)
        target3d[2, 3, 1] = True
        pred3d = np.zeros((6, 6, 6), dtype=bool)
        pred3d[2, 3, 3] = True
        res3d = binary_metrics(pred3d, target3d)
        self.assertAlmostEqual(res3d["asd"], 2.0, places=6)
        self.assertAlmostEqual(res3d["hd95"], 2.0, places=6)

        # Asymmetric shape test with analytical formula:
        # target is 2x2 square [2:4, 2:4] (4 surface voxels)
        # pred is 1x1 voxel at (5, 2) (1 surface voxel)
        # distances:
        # pred to target: d((5, 2), (3, 2)) = 2.0
        # target to pred:
        #   (3, 2) -> (5, 2) = 2.0
        #   (3, 3) -> (5, 2) = sqrt((5-3)^2 + (2-3)^2) = sqrt(5)
        #   (2, 2) -> (5, 2) = 3.0
        #   (2, 3) -> (5, 2) = sqrt((5-2)^2 + (2-3)^2) = sqrt(10)
        # ASD = (2.0 + 2.0 + sqrt(5) + 3.0 + sqrt(10)) / 5
        target_asym = np.zeros((8, 8), dtype=bool)
        target_asym[2:4, 2:4] = True
        pred_asym = np.zeros((8, 8), dtype=bool)
        pred_asym[5, 2] = True
        expected_asd = (7.0 + math.sqrt(5) + math.sqrt(10)) / 5.0
        res_asym = binary_metrics(pred_asym, target_asym)
        self.assertAlmostEqual(res_asym["asd"], expected_asd, places=6)

    def test_asd_physical_spacing_scaling(self):
        target = np.zeros((10, 10), dtype=bool)
        target[3, 3:6] = True
        pred = np.zeros((10, 10), dtype=bool)
        pred[5, 3:6] = True

        res_unit = binary_metrics(pred, target, spacing=(1.0, 1.0))
        res_double = binary_metrics(pred, target, spacing=(2.0, 2.0))
        self.assertAlmostEqual(res_unit["asd"], 2.0, places=6)
        self.assertAlmostEqual(res_double["asd"], 4.0, places=6)
        self.assertAlmostEqual(res_double["asd"], 2.0 * res_unit["asd"], places=6)

        # Anisotropic spacing test:
        # Translation is along axis 0 by 2 voxels.
        # With spacing=(3.5, 1.0), distance along axis 0 is 2 * 3.5 = 7.0
        res_aniso = binary_metrics(pred, target, spacing=(3.5, 1.0))
        self.assertAlmostEqual(res_aniso["asd"], 7.0, places=6)

    def test_hd95_uses_spacing_in_array_axis_order(self):
        target = np.zeros((5, 5, 5), dtype=bool)
        target[2, 2, 2] = True
        for axis, distance in enumerate((2.0, 3.0, 4.0)):
            prediction = np.zeros_like(target)
            index = [2, 2, 2]
            index[axis] += 1
            prediction[tuple(index)] = True
            with self.subTest(axis=axis):
                self.assertEqual(binary_metrics(prediction, target)['hd95'], 1)
                self.assertEqual(binary_metrics(prediction, target, (2, 3, 4))['hd95'], distance)

    def test_metrics_do_not_modify_inputs_or_union_masks(self):
        prediction = np.array([[0, 2, 3], [1, 3, 0]], dtype=np.uint8)
        target = np.array([[0, 3, 3], [1, 2, 0]], dtype=np.uint8)
        saved_prediction, saved_target = prediction.copy(), target.copy()
        calculate_metric_percase(prediction, target)
        binary_metrics(prediction >= 2, target >= 2)
        binary_metrics(prediction > 0, target > 0)
        np.testing.assert_array_equal(prediction, saved_prediction)
        np.testing.assert_array_equal(target, saved_target)

    def test_invalid_shapes_and_spacing_are_rejected(self):
        mask = np.ones((2, 3), dtype=bool)
        with self.assertRaises(ValueError):
            binary_metrics(mask, mask.T)
        for spacing in ((1,), (1, 0), (1, -1), (1, np.nan)):
            with self.subTest(spacing=spacing), self.assertRaises(ValueError):
                binary_metrics(mask, mask, spacing)


