import numpy as np
from scipy.ndimage import label
from scipy.spatial import ConvexHull


def extract_instances(mask_prob, threshold=0.5):
    mask = np.asarray(mask_prob) > threshold
    instances, _ = label(mask)
    return instances.astype(np.int32)


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
