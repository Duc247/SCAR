import tempfile
import unittest
from pathlib import Path

from training.config.config_utils import generate_run_dir


class RunDirNamingTests(unittest.TestCase):
    def test_run_dir_format_with_seed_and_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = generate_run_dir(
                run_root=tmp_dir,
                model_name="CMSPA-Net",
                seed=1234,
                timestamp="17h05",
            )
            self.assertEqual(run_dir.name, "CMSPA-Net_seed1234_17h05")
            self.assertTrue(run_dir.is_dir())

    def test_run_dir_format_default_timestamp_pattern(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = generate_run_dir(
                run_root=tmp_dir,
                model_name="M3-DPF",
                seed=42,
            )
            self.assertTrue(run_dir.name.startswith("M3-DPF_seed42_"))
            time_part = run_dir.name.split("_")[-1]
            self.assertIn("h", time_part)
            self.assertEqual(len(time_part), 5)

    def test_run_dir_format_seed_prefixed(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = generate_run_dir(
                run_root=tmp_dir,
                model_name="CMSPA-Net",
                seed="seed999",
                timestamp="12h30",
            )
            self.assertEqual(run_dir.name, "CMSPA-Net_seed999_12h30")

    def test_run_dir_collision_handling(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            dir1 = generate_run_dir(
                run_root=tmp_dir,
                model_name="CMSPA-Net",
                seed=1234,
                timestamp="17h05",
            )
            (dir1 / "best.pth").write_text("dummy")

            dir2 = generate_run_dir(
                run_root=tmp_dir,
                model_name="CMSPA-Net",
                seed=1234,
                timestamp="17h05",
            )
            self.assertEqual(dir2.name, "CMSPA-Net_seed1234_17h05_01")
            self.assertTrue(dir2.is_dir())


if __name__ == "__main__":
    unittest.main()
