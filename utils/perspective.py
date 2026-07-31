"""
utils/perspective.py

Homography (perspective) transform utilities.

Maps a quadrilateral drawn on the road in the raw camera frame onto a
flat rectangle ("bird's-eye view"). Once a point is in bird's-eye
space, distances in that space are proportional to real-world
distances -- which is what makes speed estimation possible.
"""

import cv2
import numpy as np


class PerspectiveTransformer:
    """
    source_points must be exactly 4 (x, y) pixel coordinates, in this
    order:
        1. Top-left
        2. Top-right
        3. Bottom-right
        4. Bottom-left

    output_width / output_height define the resolution of the warped
    space. These are NOT real-world units -- they're just how much
    "canvas" the warp is drawn on. Real-world scale is handled
    separately (see calibrate_scale.py + SpeedEstimator's
    meters_per_unit).
    """

    def __init__(self, source_points, output_width, output_height):
        self.output_width = output_width
        self.output_height = output_height

        self.source_points = np.asarray(source_points, dtype=np.float32)
        if self.source_points.shape != (4, 2):
            raise ValueError(
                "PerspectiveTransformer expects exactly 4 source points, "
                f"got shape {self.source_points.shape}."
            )

        self.destination_points = np.float32([
            [0, 0],
            [output_width, 0],
            [output_width, output_height],
            [0, output_height],
        ])

        self.matrix = cv2.getPerspectiveTransform(
            self.source_points, self.destination_points
        )
        self.inverse_matrix = cv2.getPerspectiveTransform(
            self.destination_points, self.source_points
        )

    def transform(self, points):
        """Image pixel(s) -> bird's-eye coordinate(s)."""
        points = np.asarray(points, dtype=np.float32)
        if len(points) == 0:
            return np.empty((0, 2), dtype=np.float32)
        points = points.reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(points, self.matrix)
        return transformed.reshape(-1, 2)

    def inverse_transform(self, points):
        """Bird's-eye coordinate(s) -> image pixel(s). Handy for debug overlays."""
        points = np.asarray(points, dtype=np.float32)
        if len(points) == 0:
            return np.empty((0, 2), dtype=np.float32)
        points = points.reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(points, self.inverse_matrix)
        return transformed.reshape(-1, 2)
