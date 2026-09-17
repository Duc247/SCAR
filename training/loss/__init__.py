from training.loss.losses import (
    AnatomicalSegmentationLoss,
    ClinicalAnatomicalInclusionLoss,
    DiceLoss,
    SegmentationLoss,
)


def build_loss(loss_name="ce_dice", **kwargs):
    """Build the M0-M3 compound loss or anatomical loss, returning loss and logging components."""
    normalized = loss_name.lower().replace("-", "_")
    if "num_classes" in kwargs:
        classes = kwargs.pop("num_classes")
        if "n_classes" in kwargs and kwargs["n_classes"] != classes:
            raise ValueError("Conflicting num_classes and n_classes")
        kwargs["n_classes"] = classes

    if normalized in {"anatomical", "clinical_anatomical", "anatomical_loss", "anatomical_ce_dice"}:
        return AnatomicalSegmentationLoss(**kwargs)
    elif normalized in {"ce_dice", "segmentation", "segmentation_loss"}:
        return SegmentationLoss(**kwargs)
    raise ValueError(f"Unknown loss {loss_name!r}; expected 'ce_dice' or 'anatomical'.")


__all__ = [
    "DiceLoss",
    "SegmentationLoss",
    "ClinicalAnatomicalInclusionLoss",
    "AnatomicalSegmentationLoss",
    "build_loss",
]

