from monai import transforms


def get_augmentation_preset(preset: str):
    """Get augmentation transforms for a given preset level.

    Args:
        preset: One of 'none', 'light', 'medium', 'high'

    Returns:
        List of MONAI transforms
    """
    if preset is None or preset == 'none':
        return []

    if preset == 'light':
        return []  # TODO: implement light augmentation

    if preset == 'medium':
        return []  # TODO: implement medium augmentation

    if preset == 'high':
        return []  # TODO: implement high augmentation

    raise ValueError(f"Unknown augmentation preset: {preset}. Choose from: 'none', 'light', 'medium', 'high'")


# =============================================================================
# Reference: Previous augmentation settings from train.py
# =============================================================================

# CONFIG['augmentation'] = {
#     'gaussian_std': 0.02,
#     'shift_offsets': 0.02,
#     'contrast_gamma': (0.98, 1.02),
#     'scale_factors': 0.02,
#     'rotate90_prob': 0.5,
#     'flip_prob': 0.5,
#     'spatial_pad_size': [168, 304, 304],
#     'spatial_crop_size': [160, 288, 288],
#     'rotation_range': 0.33,
#     'rotation_prob': 0.25,
#     'zoom_range': (0.9, 1.1),
#     'zoom_prob': 0.25,
# }

# Previously used transforms:
# transforms.RandGaussianNoised(keys='patch', dtype=torch.float16, prob=1, std=aug['gaussian_std']),
# transforms.RandShiftIntensityd(keys='patch', offsets=aug['shift_offsets'], safe=True, prob=1),
# transforms.RandAdjustContrastd(keys="patch", gamma=aug['contrast_gamma'], prob=1),
# transforms.RandScaleIntensityd(keys="patch", factors=aug['scale_factors'], prob=1),
# transforms.RandRotate90d(keys=["patch", "label"], prob=aug['rotate90_prob'], spatial_axes=[1,2]),
# transforms.RandFlipd(keys=['patch', 'label'], prob=aug['flip_prob'], spatial_axis=[0,1,2]),
# transforms.SpatialPadd(keys=['patch', 'label'], spatial_size=aug['spatial_pad_size'], mode='reflect'),
# transforms.RandSpatialCropd(keys=['patch', 'label'], roi_size=aug['spatial_crop_size'], random_center=True),
# transforms.RandRotated(keys=['patch', 'label'], range_x=aug['rotation_range'], range_y=aug['rotation_range'], range_z=aug['rotation_range'], prob=aug['rotation_prob'], mode=['trilinear', 'nearest']),
# transforms.RandZoomd(keys=['patch', 'label'], min_zoom=aug['zoom_range'][0], max_zoom=aug['zoom_range'][1], prob=aug['zoom_prob'], mode=['trilinear', 'nearest']),
