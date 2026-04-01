"""Tests for the dataset preparation module."""

import pytest
import torch
from PIL import Image

from diffusion_sketch.data.dataset import SketchColorDataset


class TestSketchColorDatasetInit:
    """Tests for dataset initialization and file discovery."""

    def test_discovers_png_and_jpg(self, tmp_paired_dataset):
        ds = SketchColorDataset(str(tmp_paired_dataset), image_size=64)
        assert len(ds) == 8

    def test_raises_on_empty_directory(self, empty_dir):
        with pytest.raises(FileNotFoundError, match="No images found"):
            SketchColorDataset(str(empty_dir))

    def test_single_image(self, single_image_dir):
        ds = SketchColorDataset(str(single_image_dir), image_size=64)
        assert len(ds) == 1

    def test_ignores_non_image_files(self, tmp_path):
        (tmp_path / "readme.txt").write_text("not an image")
        (tmp_path / "data.csv").write_text("a,b,c")
        img = Image.new("RGB", (512, 256))
        img.save(str(tmp_path / "valid.png"))
        ds = SketchColorDataset(str(tmp_path), image_size=64)
        assert len(ds) == 1

    def test_repr_contains_useful_info(self, tmp_paired_dataset):
        ds = SketchColorDataset(str(tmp_paired_dataset), image_size=128)
        r = repr(ds)
        assert "SketchColorDataset" in r
        assert "n_images=8" in r
        assert "size=128" in r


class TestSketchColorDatasetItems:
    """Tests for __getitem__ output shapes, types, and value ranges."""

    def test_output_shapes(self, tmp_paired_dataset):
        ds = SketchColorDataset(str(tmp_paired_dataset), image_size=64)
        sketch, target = ds[0]
        assert sketch.shape == (3, 64, 64)
        assert target.shape == (3, 64, 64)

    def test_output_dtype(self, tmp_paired_dataset):
        ds = SketchColorDataset(str(tmp_paired_dataset), image_size=64)
        sketch, target = ds[0]
        assert sketch.dtype == torch.float32
        assert target.dtype == torch.float32

    def test_normalized_range(self, tmp_paired_dataset):
        ds = SketchColorDataset(str(tmp_paired_dataset), image_size=64, augment=False)
        sketch, target = ds[0]
        assert sketch.min() >= -1.0 - 1e-5
        assert sketch.max() <= 1.0 + 1e-5
        assert target.min() >= -1.0 - 1e-5
        assert target.max() <= 1.0 + 1e-5

    def test_different_sizes(self, tmp_paired_dataset):
        for sz in [32, 128, 256]:
            ds = SketchColorDataset(str(tmp_paired_dataset), image_size=sz)
            sketch, target = ds[0]
            assert sketch.shape == (3, sz, sz)
            assert target.shape == (3, sz, sz)

    def test_all_items_loadable(self, tmp_paired_dataset):
        ds = SketchColorDataset(str(tmp_paired_dataset), image_size=64)
        for i in range(len(ds)):
            sketch, target = ds[i]
            assert sketch.shape[0] == 3
            assert target.shape[0] == 3


class TestSketchColorDatasetAugmentation:
    """Tests for augmentation behavior."""

    def test_no_augment_deterministic(self, tmp_paired_dataset):
        ds = SketchColorDataset(str(tmp_paired_dataset), image_size=64, augment=False)
        s1, t1 = ds[0]
        s2, t2 = ds[0]
        assert torch.allclose(s1, s2)
        assert torch.allclose(t1, t2)

    def test_target_always_deterministic(self, tmp_paired_dataset):
        ds = SketchColorDataset(str(tmp_paired_dataset), image_size=64, augment=True)
        _, t1 = ds[0]
        _, t2 = ds[0]
        assert torch.allclose(t1, t2)


class TestSketchColorDatasetEdgeCases:
    """Edge-case and robustness tests."""

    def test_wide_image_split(self, tmp_path):
        img = Image.new("RGB", (1024, 128), color=(100, 150, 200))
        img.save(str(tmp_path / "wide.png"))
        ds = SketchColorDataset(str(tmp_path), image_size=64)
        sketch, target = ds[0]
        assert sketch.shape == (3, 64, 64)
        assert target.shape == (3, 64, 64)

    def test_square_image_split(self, tmp_path):
        img = Image.new("RGB", (256, 256), color=(50, 100, 150))
        img.save(str(tmp_path / "square.png"))
        ds = SketchColorDataset(str(tmp_path), image_size=64)
        sketch, target = ds[0]
        assert sketch.shape == (3, 64, 64)

    def test_rgba_image_converted_to_rgb(self, tmp_path):
        img = Image.new("RGBA", (512, 256), color=(100, 150, 200, 255))
        img.save(str(tmp_path / "rgba.png"))
        ds = SketchColorDataset(str(tmp_path), image_size=64)
        sketch, target = ds[0]
        assert sketch.shape == (3, 64, 64)

    def test_grayscale_image_converted_to_rgb(self, tmp_path):
        img = Image.new("L", (512, 256), color=128)
        img.save(str(tmp_path / "gray.png"))
        ds = SketchColorDataset(str(tmp_path), image_size=64)
        sketch, target = ds[0]
        assert sketch.shape == (3, 64, 64)

    def test_jpeg_extension(self, tmp_path):
        img = Image.new("RGB", (512, 256))
        img.save(str(tmp_path / "test.jpeg"))
        ds = SketchColorDataset(str(tmp_path), image_size=64)
        assert len(ds) == 1
