"""Measurements in original-image pixels."""
from math import hypot


def point_distance(point_a, point_b):
    """Return Euclidean distance between two (x, y) points."""
    return hypot(point_a[0] - point_b[0], point_a[1] - point_b[1])


def polygon_mask(polygon, image_size):
    """Rasterize a CVAT polygon at original resolution; size is (width, height)."""
    import numpy as np
    from PIL import Image, ImageDraw

    mask = Image.new("1", image_size)
    ImageDraw.Draw(mask).polygon([tuple(point) for point in polygon], fill=1)
    return np.asarray(mask, dtype=bool)
