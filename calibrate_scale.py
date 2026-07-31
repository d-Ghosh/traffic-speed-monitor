"""
calibrate_scale.py

Step 2 of calibration (run AFTER calibrate_perspective.py).

Click two reference points a known real-world distance apart (e.g.
front bumper -> rear bumper of a car) on one or more vehicles. This
computes meters_per_unit for the chosen lane and writes it straight
into config.yaml.

Usage:
    python calibrate_scale.py --lane left_lane
    python calibrate_scale.py --lane right_lane

Then, for each vehicle you click:
    1. Type its real-world reference length in meters when prompted
       (press Enter to reuse the last value).
    2. Click the FRONT point, then the REAR point, on the video window.
    3. Repeat for more vehicles (more samples = more reliable).
    4. Press 'q' in the video window when done -- a summary prints and
       config.yaml is updated with the median scale.

Reference lengths for convenience:
    compact car     ~ 4.2 m
    sedan           ~ 4.5 m
    SUV / large car ~ 5.0 m
    semi-trailer     ~ 16.5 m
"""

import argparse
import math
from statistics import median

import cv2
import yaml

from utils.perspective import PerspectiveTransformer

CONFIG_PATH = "config.yaml"
DEFAULT_REF_LENGTH_M = 4.5
OUTLIER_THRESHOLD_FRACTION = 0.35  # samples deviating from median by more than this are excluded


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, sort_keys=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", choices=["left_lane", "right_lane"], required=True)
    args = parser.parse_args()

    config = load_config()
    lane_cfg = config["perspective"][args.lane]

    if not lane_cfg.get("source_points"):
        raise SystemExit(
            f"'{args.lane}' has no source_points yet -- run calibrate_perspective.py first."
        )

    transformer = PerspectiveTransformer(
        lane_cfg["source_points"], lane_cfg["output_width"], lane_cfg["output_height"]
    )

    cap = cv2.VideoCapture(config["video"]["path"])
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise SystemExit("Could not read a frame from the video.")

    samples = []          # list of scale_factor (meters_per_unit) values
    points = []            # current click pair
    current_ref_length = DEFAULT_REF_LENGTH_M

    def prompt_ref_length():
        nonlocal current_ref_length
        raw = input(
            f"\nReal-world length in meters for the NEXT vehicle "
            f"(Enter to reuse {current_ref_length}m): "
        ).strip()
        if raw:
            try:
                current_ref_length = float(raw)
            except ValueError:
                print("Invalid number, keeping previous value.")
        print(f"Using {current_ref_length} m. Click FRONT point, then REAR point.")

    def on_click(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        points.append((x, y))
        label = "FRONT" if len(points) == 1 else "REAR"
        print(f"  {label} point: ({x}, {y})")

        cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
        if len(points) == 2:
            cv2.line(frame, points[0], points[1], (0, 255, 0), 2)
        cv2.imshow("Scale Calibration", frame)

        if len(points) == 2:
            record_sample()

    def record_sample():
        front_px, rear_px = points
        warped = transformer.transform([front_px, rear_px])
        warped_distance = math.dist(warped[0], warped[1])

        if warped_distance == 0:
            print("  Warped distance is 0 -- points too close, or outside the calibrated zone. Discarded.")
        else:
            meters_per_unit = current_ref_length / warped_distance
            samples.append(meters_per_unit)
            print(f"  -> warped_distance={warped_distance:.3f} units, "
                  f"meters_per_unit={meters_per_unit:.5f}  (sample #{len(samples)})")

        points.clear()
        prompt_ref_length()

    cv2.namedWindow("Scale Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Scale Calibration", 1400, 800)
    cv2.setMouseCallback("Scale Calibration", on_click)
    cv2.imshow("Scale Calibration", frame)

    print(f"Calibrating lane: {args.lane}")
    print("Click multiple vehicles for a more reliable result. Press 'q' when done.")
    prompt_ref_length()

    while True:
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cv2.destroyAllWindows()

    if not samples:
        print("\nNo samples collected -- config.yaml not changed.")
        return

    med = median(samples)
    good = [s for s in samples if med and abs(s - med) / med <= OUTLIER_THRESHOLD_FRACTION]
    final_value = median(good) if good else med

    print("\n" + "=" * 50)
    print(f"Collected {len(samples)} sample(s), {len(good)} used after outlier rejection")
    print(f"meters_per_unit for {args.lane}: {final_value:.5f}")
    print("=" * 50)

    lane_cfg["meters_per_unit"] = float(final_value)
    save_config(config)
    print(f"Saved to config.yaml -> perspective.{args.lane}.meters_per_unit\n")


if __name__ == "__main__":
    main()
