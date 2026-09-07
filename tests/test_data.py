"""Semantic and integration checks using generated, non-patient data."""

from pathlib import Path
from unittest.mock import patch

import h5py
import nibabel as nib
import numpy as np
import pytest
import torch

from training.dataset.data_contract import (
    CANONICAL_LABEL_ORDER, canonicalize_label, ensure_patient_splits,
    patient_id, read_split_names, validate_patient_splits,
)
from training.dataset.myops_dataset import Myops_dataset, RandomGenerator, ResizeGenerator, random_rot_flip
from preprocessing.preprocessing import discover_cases, load_aligned_case, normalize_image
from preprocessing.process_and_save import preprocess_dataset


def write_lists(directory, **splits):
    directory.mkdir(parents=True, exist_ok=True)
    for split, names in splits.items():
        (directory / f"{split}.txt").write_text("".join(f"{name}\n" for name in names), encoding="utf-8")


def npz_fixture(tmp_path, metadata=True):
    roots = [tmp_path / name for name in ("cine", "lge", "t2w")]
    image = np.arange(16, dtype=np.float32).reshape(4, 4) / 16
    label = np.tile(np.arange(4, dtype=np.uint8), (4, 1))
    extra = {"label_order": CANONICAL_LABEL_ORDER} if metadata else {}
    for root in roots:
        root.mkdir()
        np.savez(root / "case0001_slice000.npz", image=image, label=label, **extra)
    lists = tmp_path / "lists"
    write_lists(lists, train=["case0001_slice000"])
    return roots, lists, image, label


def test_legacy_class_remap_is_non_mutating_and_explicit():
    source = np.asarray([0, 1, 2, 3], dtype=np.uint8)
    np.testing.assert_array_equal(canonicalize_label(source, "legacy"), [0, 1, 3, 2])
    np.testing.assert_array_equal(canonicalize_label(source, "canonical"), source)
    np.testing.assert_array_equal(source, [0, 1, 2, 3])
    for bad in ([0, 4], [1, 2.5], [0, np.nan], [-1, 0]):
        with pytest.raises(ValueError):
            canonicalize_label(np.asarray(bad), "canonical")


def test_split_by_patient_preserves_test_and_source_and_resume(tmp_path):
    source, output = tmp_path / "original", tmp_path / "run"
    train = [f"case{i:04d}_slice{j:03d}" for i in range(1, 11) for j in range(i % 3 + 1)]
    write_lists(source, train=train, test_vol=["case0011", "case0012"])
    before = (source / "train.txt").read_bytes()
    result = ensure_patient_splits(source, seed=11, output_dir=output)
    assert result == output
    splits = {name: read_split_names(result, name) for name in ("train", "val", "test_vol")}
    patient_sets = validate_patient_splits(splits)
    assert len(patient_sets["val"]) == 2
    assert set(splits["train"] + splits["val"]) == set(train)
    assert splits["test_vol"] == ["case0011", "case0012"]
    assert (source / "train.txt").read_bytes() == before
    second = ensure_patient_splits(source, seed=11, output_dir=tmp_path / "run2")
    assert (second / "val.txt").read_bytes() == (output / "val.txt").read_bytes()
    # A saved run keeps its exact split, even after source manifests change.
    write_lists(source, train=["different_slice000"])
    assert ensure_patient_splits(source, seed=99, output_dir=output) == output
    assert read_split_names(output, "val") == splits["val"]


def test_split_rejects_patient_leakage_and_duplicates(tmp_path):
    with pytest.raises(ValueError, match="Patient leakage"):
        validate_patient_splits({"train": ["case001_slice000"], "val": ["case001_slice001"]})
    write_lists(tmp_path, train=["case001_slice000", "case001_slice000"])
    with pytest.raises(ValueError, match="Duplicate"):
        read_split_names(tmp_path, "train")
    for name in ("../case", "..\\case", "C:case"):
        with pytest.raises(ValueError):
            patient_id(name)


def test_dataset_metadata_prevents_double_remap(tmp_path):
    roots, lists, image, label = npz_fixture(tmp_path)
    dataset = Myops_dataset(*roots, lists, "train", transform=ResizeGenerator((8, 8)))
    sample = dataset[0]
    assert sample["image"].shape == (1, 8, 8)
    assert sample["image"].dtype == torch.float32
    assert sample["label"].dtype == torch.int64
    assert set(sample["label"].unique().tolist()) == {0, 1, 2, 3}
    assert sample["label"][0].tolist() == [0, 0, 1, 1, 2, 2, 3, 3]
    assert not sample["has_geometry"]
    assert np.isnan(sample["spacing"]).all()
    with pytest.raises(ValueError, match="conflicts with metadata"):
        Myops_dataset(*roots, lists, "train", label_order="legacy")[0]


def test_dataset_legacy_warning_and_canonical_override(tmp_path):
    roots, lists, _, label = npz_fixture(tmp_path, metadata=False)
    with pytest.warns(UserWarning, match="legacy I-MMSeg"):
        legacy = Myops_dataset(*roots, lists, "train")[0]
    np.testing.assert_array_equal(legacy["label"][0], [0, 1, 3, 2])
    np.testing.assert_array_equal(Myops_dataset(*roots, lists, "train", label_order="canonical")[0]["label"], label)


def test_dataset_rejects_missing_modality_and_disagreeing_masks(tmp_path):
    roots, lists, image, label = npz_fixture(tmp_path)
    path = roots[2] / "case0001_slice000.npz"
    path.unlink()
    with pytest.raises(FileNotFoundError, match="aligned modality"):
        Myops_dataset(*roots, lists, "train")
    np.savez(path, image=image, label=np.zeros_like(label), label_order=CANONICAL_LABEL_ORDER)
    with pytest.raises(ValueError, match="target masks differ"):
        Myops_dataset(*roots, lists, "train")[0]


def test_synchronized_spatial_transforms_and_nonwrapping_translation():
    image = np.zeros((16, 16), dtype=np.float32)
    image[3:5, 15] = 1
    arrays = random_rot_flip(image, image, image, image.astype(np.uint8))
    for transformed in arrays[1:]:
        np.testing.assert_array_equal(arrays[0], transformed)
    sample = {"image": image, "image1": image, "image2": image, "label": image.astype(np.uint8)}
    # Skip rotations/intensity changes; translate right one pixel beyond edge.
    with patch("training.dataset.myops_dataset.random.random", side_effect=[1., 1., 0.] + [1.] * 9), patch("training.dataset.myops_dataset.random.randint", side_effect=[0, 1]):
        transformed = RandomGenerator((16, 16))(sample)
    assert transformed["label"].sum() == 0
    assert transformed["image"].sum() == 0
    for key in ("image1", "image2"):
        torch.testing.assert_close(transformed["image"], transformed[key])


def raw_fixture(tmp_path, count=5):
    source = tmp_path / "raw"
    affine = np.diag([1.25, 1.5, 7.0, 1.0])
    affine[:3, 3] = [12, -5, 23]
    image = np.arange(8 * 8 * 2, dtype=np.float32).reshape(8, 8, 2)
    label = (image.astype(np.uint8) % 4)
    for directory in ("bSSFP", "LGE", "T2w", "label"):
        (source / directory).mkdir(parents=True)
        for i in range(count):
            case = f"case{i + 1:04d}"
            suffix = "_gt" if directory == "label" else ""
            data = label if directory == "label" else image
            volume = nib.Nifti1Image(data, affine)
            volume.header.set_xyzt_units("mm")
            nib.save(volume, source / directory / f"{case}{suffix}.nii.gz")
    return source, image, label, affine


def test_preprocess_to_loader_roundtrip_geometry_and_patient_splits(tmp_path):
    source, image, label, affine = raw_fixture(tmp_path)
    test_list = tmp_path / "held_out" / "test_vol.txt"
    write_lists(test_list.parent, test_vol=["case0005"])
    output = tmp_path / "processed"
    info = preprocess_dataset(source, output, test_list=test_list, show_progress=False)
    assert info["patients"]["test_vol"] == ["case0005"]
    copied = ensure_patient_splits(output / "lists", output_dir=tmp_path / "run_splits")
    assert read_split_names(copied, "val_vol") == info["patients"]["val"]
    splits = {split: read_split_names(output / "lists", split) for split in ("train", "val", "test_vol")}
    validate_patient_splits(splits)
    roots = [output / mod / "test_vol_h5" for mod in ("bSSFP", "LGE", "T2w")]
    sample = Myops_dataset(*roots, output / "lists", "test_vol")[0]
    assert sample["has_geometry"]
    np.testing.assert_array_equal(sample["spacing"], [1.25, 1.5, 7])
    np.testing.assert_array_equal(sample["affine"], affine)
    np.testing.assert_allclose(sample["image"], image / 255)
    np.testing.assert_array_equal(sample["label"], canonicalize_label(label, "legacy"))
    assert sample["image"].shape == (8, 8, 2)
    # Source labels are unchanged; preprocessing is strictly one-way packaging.
    np.testing.assert_array_equal(nib.load(source / "label" / "case0005_gt.nii.gz").get_fdata(), label)
    with pytest.raises(FileExistsError, match="overwrite"):
        preprocess_dataset(source, output, show_progress=False)


def test_preprocessing_refuses_mismatched_ids_and_geometry(tmp_path):
    source, image, _, affine = raw_fixture(tmp_path, count=1)
    wrong_affine = affine.copy()
    wrong_affine[0, 3] += 3
    nib.save(nib.Nifti1Image(image, wrong_affine), source / "LGE" / "case0001.nii.gz")
    with pytest.raises(ValueError, match="Unaligned shape/affine"):
        load_aligned_case(discover_cases(source)["case0001"])
    (source / "LGE" / "case0001.nii.gz").rename(source / "LGE" / "case9999.nii.gz")
    with pytest.raises(ValueError, match="Patient IDs differ"):
        discover_cases(source)


def test_normalization_is_explicit_finite_and_bounded():
    np.testing.assert_allclose(normalize_image(np.asarray([0, 127.5, 255])), [0, 0.5, 1])
    np.testing.assert_array_equal(normalize_image(np.zeros((4, 4)), "percentile"), 0)
    with pytest.raises(ValueError, match="release intensities"):
        normalize_image(np.asarray([0, 4095]))
    with pytest.raises(ValueError, match="finite"):
        normalize_image(np.asarray([np.nan]))


def test_unknown_nifti_units_preserved_without_claiming_mm(tmp_path):
    source, _, _, _ = raw_fixture(tmp_path, count=3)
    for path in source.rglob("*.nii.gz"):
        volume = nib.load(path)
        volume.header.set_xyzt_units("unknown")
        nib.save(volume, path)
    output = tmp_path / "processed"
    preprocess_dataset(source, output, show_progress=False)
    roots = [output / mod / "test_vol_h5" for mod in ("bSSFP", "LGE", "T2w")]
    sample = Myops_dataset(*roots, output / "lists", "test_vol")[0]
    assert not sample["has_geometry"]
    assert sample["spacing_unit"] == "unknown"
    assert np.isfinite(sample["affine"]).all()
    assert sample["has_affine"]


@pytest.mark.parametrize('unit,scale', [('meter', 1000.), ('micron', .001)])
def test_physical_unit_conversion_preserves_mm_coordinates(tmp_path, unit, scale):
    source, _, _, expected_affine = raw_fixture(tmp_path, count=3)
    for path in source.rglob('*.nii.gz'):
        volume = nib.load(path)
        array = volume.get_fdata().astype(np.float32)
        affine = volume.affine.copy()
        affine[:3, :] /= scale
        converted = nib.Nifti1Image(array, affine)
        converted.header.set_xyzt_units(unit)
        nib.save(converted, path)
    output = tmp_path / 'converted'
    preprocess_dataset(source, output, show_progress=False)
    roots = [output / mod / 'test_vol_h5' for mod in ('bSSFP', 'LGE', 'T2w')]
    sample = Myops_dataset(*roots, output / 'lists', 'test_vol')[0]
    assert sample['has_geometry'] and sample['spacing_unit'] == 'mm'
    np.testing.assert_allclose(sample['affine'], expected_affine, atol=1e-5)
    np.testing.assert_allclose(sample['spacing'], [1.25, 1.5, 7.], atol=1e-5)


def test_cached_validation_volume_reassembly_and_gaps(tmp_path):
    roots = [tmp_path / m / 'train_npz' for m in ('bSSFP', 'LGE', 'T2w')]
    label = np.tile(np.arange(4, dtype=np.uint8), (4, 1))
    for root in roots:
        root.mkdir(parents=True)
        for i in range(3):
            np.savez(root / f'case01_slice{i:03d}.npz', image=np.full((4, 4), i / 3, dtype=np.float32),
                     label=label, label_order=CANONICAL_LABEL_ORDER)
    lists = tmp_path / 'lists'
    write_lists(lists, val_vol=['case01'])
    volume_roots = [p.parent / 'val_vol_h5' for p in roots]
    sample = Myops_dataset(*volume_roots, lists, 'val_vol')[0]
    assert sample['image'].shape == (4, 4, 3)
    np.testing.assert_allclose(sample['image'][0, 0], [0, 1/3, 2/3])
    assert not sample['has_affine'] and not sample['has_geometry']
    (roots[0] / 'case01_slice001.npz').unlink()
    with pytest.raises(ValueError, match='contiguous'):
        Myops_dataset(*volume_roots, lists, 'val_vol')


def test_rare_sampler_reads_unaugmented_canonical_labels(tmp_path):
    from training.dataset.sampler import build_rare_class_sampler
    roots, lists, _, _ = npz_fixture(tmp_path)
    def forbidden_transform(sample):
        raise AssertionError('Sampler must not execute random augmentation')
    dataset = Myops_dataset(*roots, lists, 'train', transform=forbidden_transform)
    sampler = build_rare_class_sampler(dataset, rare_boost=3)
    assert sampler.weights.tolist() == [3.]
    assert list(sampler) == [0]
    with pytest.raises(ValueError):
        build_rare_class_sampler(dataset, rare_boost=float('nan'))


def test_build_splits_reuses_manifests_and_refuses_mismatched_fixed_test(tmp_path):
    from preprocessing.build_splits import build_splits
    source, _, _, _ = raw_fixture(tmp_path)
    processed = tmp_path / 'cache'
    info = preprocess_dataset(source, processed, show_progress=False)
    lists = processed / 'lists'
    old_train = (lists / 'train.txt').read_bytes()
    result = build_splits(processed, lists, seed=999)
    assert result['sample_counts']['train'] == info['sample_counts']['train']
    assert (lists / 'train.txt').read_bytes() == old_train
    fixed = tmp_path / 'bad/test_vol.txt'
    write_lists(fixed.parent, test_vol=['unseen'])
    with pytest.raises(ValueError, match='held-out'):
        build_splits(processed, lists, test_list=fixed)
