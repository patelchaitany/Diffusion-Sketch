"""Paired sketch-color dataset for pix2pix-style image pairs.

Expects each image file to contain a side-by-side pair:
  left half  = colored target image
  right half = sketch input image
"""

import os
import glob
from PIL import Image
from torch.utils.data import Dataset

from .transforms import get_resize_transform, get_input_transform, get_target_transform


class SketchColorDataset(Dataset):
    """Loads paired sketch-color images from a directory.

    Args:
        root_dir: directory containing .png/.jpg/.jpeg image files.
        image_size: spatial size to resize both halves to.
        augment: whether to apply color jitter augmentation on sketches.
    """

    EXTENSIONS = ("*.png", "*.jpg", "*.jpeg")

    def __init__(self, root_dir: str, image_size: int = 256, augment: bool = True):
        self.root_dir = root_dir
        self.image_size = image_size

        self.files = sorted(
            f
            for ext in self.EXTENSIONS
            for f in glob.glob(os.path.join(root_dir, ext))
        )
        if not self.files:
            raise FileNotFoundError(
                f"No images found in '{root_dir}'. "
                f"Place paired images (left=color, right=sketch) with extensions "
                f"{self.EXTENSIONS}."
            )

        self.resize = get_resize_transform(image_size)
        self.input_transform = get_input_transform() if augment else get_target_transform()
        self.target_transform = get_target_transform()

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        image = Image.open(self.files[index]).convert("RGB")
        w, h = image.size

        target = image.crop((0, 0, w // 2, h))
        sketch = image.crop((w // 2, 0, w, h))

        target = self.resize(target)
        sketch = self.resize(sketch)

        sketch = self.input_transform(sketch)
        target = self.target_transform(target)

        return sketch, target

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"root='{self.root_dir}', "
            f"n_images={len(self)}, "
            f"size={self.image_size})"
        )
