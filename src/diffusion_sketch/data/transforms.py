"""Image transforms for sketch-color paired datasets."""

from torchvision import transforms


def get_resize_transform(image_size: int = 256) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
    ])


def get_input_transform() -> transforms.Compose:
    """Transform for sketch (input) images — includes augmentation."""
    return transforms.Compose([
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def get_target_transform() -> transforms.Compose:
    """Transform for color (target) images — no augmentation."""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
