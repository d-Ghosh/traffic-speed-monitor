"""
utils/speed.py

Turns a vehicle's frame-to-frame movement in bird's-eye space into a
smoothed real-world speed (km/h).

Design notes:
- Distances are measured in warped/bird's-eye "units" (whatever
  resolution PerspectiveTransformer's output_width/output_height use).
- meters_per_unit converts those units into real meters. It comes
  from calibrate_scale.py and is the ONLY place real-world scale
  enters the pipeline.
- min_distance_m / max_plausible_kmph are both expressed in real
  units so they mean the same thing regardless of warp resolution.
"""

import math


class SpeedEstimator:
    """
    One instance per lane (each lane has its own meters_per_unit).

    Call update() once per tracked vehicle per frame.
    """

    def __init__(
        self,
        fps,
        meters_per_unit,
        ema_alpha=0.3,
        min_distance_m=0.30,
        max_plausible_kmph=180,
        max_frame_gap=15,
    ):
        if fps <= 0:
            raise ValueError("fps must be > 0")
        if meters_per_unit <= 0:
            raise ValueError(
                "meters_per_unit must be > 0 -- run calibrate_scale.py first"
            )

        self.fps = fps
        self.meters_per_unit = meters_per_unit
        self.ema_alpha = ema_alpha
        self.max_plausible_kmph = max_plausible_kmph
        self.max_frame_gap = max_frame_gap

        # convert the real-world noise threshold into warped units once,
        # so the hot path only ever compares against warped-unit distances
        self.min_distance_units = min_distance_m / meters_per_unit

        self._positions = {}       # vehicle_id -> last warped (x, y)
        self._frames = {}          # vehicle_id -> last frame_index
        self._smoothed_speed = {}  # vehicle_id -> EMA speed (km/h)

    def update(self, vehicle_id, warped_position, frame_index):
        """
        warped_position: (x, y) from PerspectiveTransformer.transform()
        frame_index: current frame number (int)

        Returns the current smoothed speed estimate in km/h.
        """
        if vehicle_id not in self._positions:
            self._positions[vehicle_id] = warped_position
            self._frames[vehicle_id] = frame_index
            self._smoothed_speed[vehicle_id] = 0.0
            return 0.0

        prev_position = self._positions[vehicle_id]
        prev_frame = self._frames[vehicle_id]
        frame_gap = frame_index - prev_frame

        # always advance state, even if this particular reading gets rejected,
        # so tracking keeps going instead of freezing on a rejected sample
        self._positions[vehicle_id] = warped_position
        self._frames[vehicle_id] = frame_index

        if frame_gap <= 0 or frame_gap > self.max_frame_gap:
            return round(self._smoothed_speed[vehicle_id], 2)

        distance_units = math.dist(prev_position, warped_position)
        if distance_units < self.min_distance_units:
            # sub-threshold movement -- likely detector jitter, not real motion
            return round(self._smoothed_speed[vehicle_id], 2)

        dt = frame_gap / self.fps
        distance_m = distance_units * self.meters_per_unit
        raw_kmph = (distance_m / dt) * 3.6

        if raw_kmph > self.max_plausible_kmph:
            # implausible single-frame spike (ID switch, detector glitch)
            return round(self._smoothed_speed[vehicle_id], 2)

        previous = self._smoothed_speed[vehicle_id]
        smoothed = self.ema_alpha * raw_kmph + (1 - self.ema_alpha) * previous
        self._smoothed_speed[vehicle_id] = smoothed

        return round(smoothed, 2)

    def reset(self, vehicle_id):
        """Call if a tracker_id is confirmed to have switched to a different vehicle."""
        self._positions.pop(vehicle_id, None)
        self._frames.pop(vehicle_id, None)
        self._smoothed_speed.pop(vehicle_id, None)
