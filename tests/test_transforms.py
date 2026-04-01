"""Tests for transform functions."""

import torch
from PIL import Image

from diffusion_sketch.data.transforms import (
    get_resize_transform,
    get_input_transform,
    get_target_transform,
)


class TestResizeTransform:
    def test_resize_to_target(self):
        img = Image.new("RGB", (512, 256))
        resized = get_resize_transform(128)(img)
        assert resized.size == (128, 128)

    def test_resize_preserves_mode(self):
        img = Image.new("RGB", (100, 200))
        resized = get_resize_transform(64)(img)
        assert resized.mode == "RGB"

    def test_resize_various_sizes(self):
        img = Image.new("RGB", (300, 400))
        for sz in [32, 64, 128, 256]:
            resized = get_resize_transform(sz)(img)
            assert resized.size == (sz, sz)


class TestInputTransform:
    def test_output_is_tensor(self):
        img = Image.new("RGB", (64, 64))
        tensor = get_input_transform()(img)
        assert isinstance(tensor, torch.Tensor)

    def test_output_shape(self):
        img = Image.new("RGB", (64, 64))
        tensor = get_input_transform()(img)
        assert tensor.shape == (3, 64, 64)

    def test_output_normalized(self):
        img = Image.new("RGB", (32, 32), color=(128, 128, 128))
        torch.manual_seed(0)
        tensor = get_input_transform()(img)
        assert tensor.min() >= -1.0 - 1e-5
        assert tensor.max() <= 1.0 + 1e-5


class TestTargetTransform:
    def test_output_is_tensor(self):
        img = Image.new("RGB", (64, 64))
        tensor = get_target_transform()(img)
        assert isinstance(tensor, torch.Tensor)

    def test_output_deterministic(self):
        img = Image.new("RGB", (32, 32), color=(200, 100, 50))
        t1 = get_target_transform()(img)
        t2 = get_target_transform()(img)
        assert torch.allclose(t1, t2)

    def test_output_shape(self):
        img = Image.new("RGB", (48, 48))
        tensor = get_target_transform()(img)
        assert tensor.shape == (3, 48, 48)

    def test_output_normalized_range(self):
        img = Image.new("RGB", (32, 32), color=(0, 0, 0))
        tensor = get_target_transform()(img)
        assert torch.allclose(tensor, torch.full_like(tensor, -1.0))

        img_white = Image.new("RGB", (32, 32), color=(255, 255, 255))
        tensor_w = get_target_transform()(img_white)
        assert torch.allclose(tensor_w, torch.full_like(tensor_w, 1.0), atol=0.01)
