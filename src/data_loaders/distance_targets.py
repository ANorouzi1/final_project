import numpy as np
from scipy.ndimage import distance_transform_edt


def instance_center_distance_transform(instance_mask):
    """Unsigned per-instance distance target in [0, 1]."""
    instance_mask = np.asarray(instance_mask)
    distance = np.zeros(instance_mask.shape, dtype=np.float32)
    for instance_id in np.unique(instance_mask):
        if instance_id == 0:
            continue
        current = instance_mask == instance_id
        current_distance = distance_transform_edt(current).astype(np.float32)
        if current_distance.max() > 0:
            current_distance /= current_distance.max()
        distance = np.maximum(distance, current_distance)
    return distance


def signed_instance_distance_field(instance_mask, outside_clip=32.0):
    """Single-channel SDF target that keeps touching field instances separable.

    The output is in [-1, 1]. Pixels inside each instance are positive and are
    normalized per instance, with boundary pixels near 0. Background pixels are
    negative according to distance from the nearest field.
    """
    instance_mask = np.asarray(instance_mask)
    sdf = np.zeros(instance_mask.shape, dtype=np.float32)
    foreground = instance_mask > 0

    for instance_id in np.unique(instance_mask):
        if instance_id == 0:
            continue
        current = instance_mask == instance_id
        inside = distance_transform_edt(current).astype(np.float32) - 1.0
        inside = np.clip(inside, 0.0, None)
        if inside.max() > 0:
            inside /= inside.max()
        sdf[current] = inside[current]

    background = ~foreground
    if np.any(background):
        outside = distance_transform_edt(background).astype(np.float32)
        if outside_clip and outside_clip > 0:
            outside = np.clip(outside / float(outside_clip), 0.0, 1.0)
        elif outside.max() > 0:
            outside /= outside.max()
        sdf[background] = -outside[background]

    return sdf.astype(np.float32)
