"""Shared pytest fixtures for Diffusion-Sketch tests."""

import os
import pytest
from PIL import Image


@pytest.fixture
def tmp_paired_dataset(tmp_path):
    """Create a temporary directory with synthetic paired images.

    Each image is 512x256 (left=color, right=sketch) — the standard
    pix2pix format that SketchColorDataset expects.
    """
    n_images = 8
    img_w, img_h = 512, 256

    for i in range(n_images):
        img = Image.new("RGB", (img_w, img_h))

        for x in range(img_w // 2):
            for y in range(img_h):
                r = int(255 * x / (img_w // 2))
                g = int(255 * y / img_h)
                b = (i * 37) % 256
                img.putpixel((x, y), (r, g, b))

        for x in range(img_w // 2, img_w):
            for y in range(img_h):
                v = int(255 * (x - img_w // 2) / (img_w // 2))
                img.putpixel((x, y), (v, v, v))

        ext = "png" if i % 2 == 0 else "jpg"
        img.save(os.path.join(tmp_path, f"pair_{i:04d}.{ext}"))

    return tmp_path


@pytest.fixture
def empty_dir(tmp_path):
    """An empty directory for testing error cases."""
    empty = tmp_path / "empty"
    empty.mkdir()
    return empty


@pytest.fixture
def single_image_dir(tmp_path):
    """Directory with exactly one paired image."""
    img = Image.new("RGB", (512, 256), color=(128, 64, 32))
    img.save(os.path.join(tmp_path, "single.png"))
    return tmp_path
