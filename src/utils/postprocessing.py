import numpy as np
from scipy.ndimage import distance_transform_edt, label
from scipy.spatial import ConvexHull


def extract_instances(mask_prob, threshold=0.5):
    mask = np.asarray(mask_prob) > threshold
    instances, _ = label(mask)
    return instances.astype(np.int32)


def instances_from_mask_and_sdf(
    mask_prob,
    sdf,
    mask_threshold=0.5,
    core_threshold=0.15,
    min_core_area=4,
    min_instance_area=20,
):
    """Use the mask head as support and the SDF head as split/core evidence.

    Disagreements are resolved conservatively:

    * mask says background -> background, even if SDF is positive
    * mask says foreground and SDF is high -> confident instance core
    * mask says foreground and SDF is low -> boundary/uncertain pixel assigned
      to the nearest confident core inside the same connected mask component
    """
    mask_prob = np.asarray(mask_prob).squeeze()
    sdf = np.asarray(sdf).squeeze()
    support = mask_prob > mask_threshold
    if not np.any(support):
        return np.zeros_like(support, dtype=np.int32)

    support_labels, n_support = label(support)
    instances = np.zeros_like(support_labels, dtype=np.int32)
    next_id = 1

    for support_id in range(1, n_support + 1):
        component = support_labels == support_id
        if int(component.sum()) < min_instance_area:
            continue

        cores = component & (sdf > core_threshold)
        cores = _remove_small_components(cores, min_core_area)
        core_labels, n_cores = label(cores)

        if n_cores == 0:
            instances[component] = next_id
            next_id += 1
            continue

        _, nearest = distance_transform_edt(core_labels == 0, return_indices=True)
        expanded = core_labels[nearest[0], nearest[1]]
        expanded[~component] = 0

        for core_id in range(1, n_cores + 1):
            instance = expanded == core_id
            if int(instance.sum()) < min_instance_area:
                continue
            instances[instance] = next_id
            next_id += 1

    return instances.astype(np.int32)


def _remove_small_components(mask, min_area):
    if min_area <= 1:
        return mask
    labels, n_labels = label(mask)
    keep = np.zeros_like(mask, dtype=bool)
    for component_id in range(1, n_labels + 1):
        component = labels == component_id
        if int(component.sum()) >= min_area:
            keep |= component
    return keep


def polygonize_instances(instances, simplify=True):
    polygons = {}
    for instance_id in np.unique(instances):
        if instance_id == 0:
            continue
        ys, xs = np.where(instances == instance_id)
        if len(xs) < 3:
            continue
        points = np.stack([xs, ys], axis=1)
        if simplify and len(points) >= 4:
            try:
                hull = ConvexHull(points)
                points = points[hull.vertices]
            except Exception:
                pass
        polygons[int(instance_id)] = points.tolist()
    return polygons
