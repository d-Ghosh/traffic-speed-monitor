"""
calibrate_perspective.py

Step 1 of calibration. Click 4 points on the road for each lane to
define the homography source quadrilateral, preview the resulting
bird's-eye view, and save the points into config.yaml.

Usage:
    python calibrate_perspective.py

For each lane, click in this exact order:
    1. Top-left
    2. Top-right
    3. Bottom-right
    4. Bottom-left

Controls while clicking:
    ENTER      confirm the 4 points and preview
    BACKSPACE  undo last point
    R          reset all points for this lane
    ESC        quit without saving

After the preview, press any key to move to the next lane (or finish).
Good source points: lane markings and road edges should look roughly
vertical/parallel in the preview, and cars shouldn't look absurdly
stretched.
"""

import cv2
import numpy as np
import yaml

CONFIG_PATH = "config.yaml"


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def save_config(config):
    with open(CONFIG_PATH, "w") as f:
        yaml.dump(config, f, sort_keys=False)


def collect_points(original_frame, title):
    points = []

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 1400, 800)
    cv2.setMouseCallback(title, on_click)

    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)
    print("Click, in order: 1) Top-left  2) Top-right  3) Bottom-right  4) Bottom-left")
    print("ENTER=confirm  BACKSPACE=undo  R=reset  ESC=quit\n")

    labels = ["TL", "TR", "BR", "BL"]

    while True:
        display = original_frame.copy()
        cv2.putText(display, title, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        for i, p in enumerate(points):
            cv2.circle(display, p, 7, (0, 0, 255), -1)
            cv2.putText(display, labels[i], (p[0] + 10, p[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        if len(points) > 1:
            cv2.polylines(display, [np.array(points)], len(points) == 4, (0, 255, 0), 2)

        cv2.imshow(title, display)
        key = cv2.waitKey(1)

        if key == 27:  # ESC
            cv2.destroyAllWindows()
            raise SystemExit("Calibration cancelled.")
        elif key == 8:  # BACKSPACE
            if points:
                points.pop()
        elif key == ord("r"):
            points.clear()
        elif key == 13 and len(points) == 4:  # ENTER
            break

    cv2.destroyWindow(title)
    return points


def preview_birds_eye(original_frame, points, output_width, output_height, title):
    src = np.float32(points)
    dst = np.float32([[0, 0], [output_width, 0], [output_width, output_height], [0, output_height]])
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(original_frame, matrix, (output_width, output_height))

    for y in range(0, output_height, 100):
        cv2.line(warped, (0, y), (output_width, y), (0, 255, 0), 1)
    for x in range(0, output_width, 100):
        cv2.line(warped, (x, 0), (x, output_height), (255, 0, 0), 1)

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, 500, 900)
    cv2.imshow(title, warped)

    print("\nCheck: lane markings + road edges should look roughly vertical/parallel.")
    print("Press any key to continue...")
    cv2.waitKey(0)
    cv2.destroyWindow(title)


def main():
    config = load_config()
    video_path = config["video"]["path"]

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise SystemExit(f"Could not read a frame from '{video_path}'. Check video.path in config.yaml.")

    for lane_key, lane_title in [("left_lane", "LEFT LANE"), ("right_lane", "RIGHT LANE")]:
        lane_cfg = config["perspective"][lane_key]
        points = collect_points(frame, lane_title)
        preview_birds_eye(
            frame, points, lane_cfg["output_width"], lane_cfg["output_height"],
            f"{lane_title} - Bird's-eye preview",
        )
        lane_cfg["source_points"] = [list(p) for p in points]
        # scale is invalidated whenever the source quad changes
        lane_cfg["meters_per_unit"] = None

    save_config(config)

    print("\n" + "=" * 50)
    print("Saved source_points for both lanes to config.yaml")
    print("Next: run calibrate_scale.py to set meters_per_unit for each lane.")
    print("=" * 50)


if __name__ == "__main__":
    main()
