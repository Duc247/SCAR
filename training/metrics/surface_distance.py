"""Physical surface distance and Hausdorff (HD95) metrics."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


def binary_metrics(prediction, target, spacing=None, compute_distance=True, empty_mode="undefined"):
    """Symmetric HD95 and ASD on 1-connected surfaces; spacing follows array axis order.

    Missing surfaces have no finite distance. Report their status separately instead
    of assigning zero distance to a completely missed lesion.
    """
    pred, truth = np.asarray(prediction, dtype=bool), np.asarray(target, dtype=bool)
    if pred.shape != truth.shape:
        raise ValueError("Prediction and target shapes differ.")
    if spacing is not None:
        spacing = np.asarray(spacing, dtype=float)
        if spacing.shape != (pred.ndim,) or not np.isfinite(spacing).all() or (spacing <= 0).any():
            raise ValueError("Spacing must contain one finite positive value per array axis.")
    n_pred, n_true = int(pred.sum()), int(truth.sum())
    if n_pred == 0 and n_true == 0:
        if empty_mode == "defined":
            return {"dice": 1.0, "iou": 1.0, "hd95": 0.0, "asd": 0.0, "status": "both_empty"}
        return {"dice": None, "iou": None, "hd95": None, "asd": None, "status": "both_empty"}
    overlap = np.logical_and(pred, truth).sum()
    result = {
        "dice": float(2 * overlap / (n_pred + n_true)),
        "iou": float(overlap / (n_pred + n_true - overlap)),
        "hd95": None,
        "asd": None,
        "status": "ok",
    }
    if not n_pred or not n_true:
        if empty_mode == "defined":
            result["hd95"] = float("inf")
            result["asd"] = float("inf")
            result["status"] = "one_empty"
        else:
            result["status"] = "prediction_empty" if not n_pred else "target_empty"
        return result
    if not compute_distance:
        if spacing is None:
            result["status"] = "missing_physical_geometry"
        return result
    structure = generate_binary_structure(pred.ndim, 1)
    pred_surface = pred ^ binary_erosion(pred, structure=structure, border_value=0)
    true_surface = truth ^ binary_erosion(truth, structure=structure, border_value=0)
    distances = np.concatenate((
        distance_transform_edt(~true_surface, sampling=spacing)[pred_surface],
        distance_transform_edt(~pred_surface, sampling=spacing)[true_surface],
    ))
    result["hd95"] = float(np.percentile(distances, 95))
    result["asd"] = float(np.mean(distances))
    return result


def calculate_metric_percase(pred, gt, spacing=None):
    """Legacy tuple API, honest empty-mask behavior, no input mutation."""
    result = binary_metrics(pred, gt, spacing, empty_mode="undefined")
    distance = result["hd95"]
    if distance is None:
        distance = np.nan if result["status"] == "both_empty" else np.inf
    return (result["dice"] if result["dice"] is not None else np.nan), distance
