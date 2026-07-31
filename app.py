"""
app.py

Main pipeline: YOLO detection -> ByteTrack tracking -> per-lane
homography -> speed estimation -> annotated video window.

Run calibrate_perspective.py and calibrate_scale.py first -- this
script will refuse to start if either lane is missing calibration.
"""

import yaml
import cv2
import supervision as sv
from ultralytics import YOLO

from utils.perspective import PerspectiveTransformer
from utils.speed import SpeedEstimator

CONFIG_PATH = "config.yaml"
WEIGHTS_PATH = "weights/yolo11s.pt"
VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # COCO: car, motorcycle, bus, truck
DEBUG = False  # set True to print per-detection pixel/world coordinates


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def build_lane(lane_cfg, fps, speed_cfg):
    if not lane_cfg.get("source_points") or not lane_cfg.get("meters_per_unit"):
        raise SystemExit(
            "Missing calibration. Run calibrate_perspective.py then "
            "calibrate_scale.py for both lanes before starting app.py."
        )

    transformer = PerspectiveTransformer(
        lane_cfg["source_points"], lane_cfg["output_width"], lane_cfg["output_height"]
    )
    estimator = SpeedEstimator(
        fps=fps,
        meters_per_unit=lane_cfg["meters_per_unit"],
        ema_alpha=speed_cfg["ema_alpha"],
        min_distance_m=speed_cfg["min_distance_m"],
        max_plausible_kmph=speed_cfg["max_plausible_kmph"],
        max_frame_gap=speed_cfg["max_frame_gap"],
    )
    return transformer, estimator


def main():
    config = load_config()
    speed_cfg = config["speed_estimation"]

    cap = cv2.VideoCapture(config["video"]["path"])
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25  # fallback if the container doesn't report fps

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    lane_split_x = frame_width // 2

    transformer_left, estimator_left = build_lane(config["perspective"]["left_lane"], fps, speed_cfg)
    transformer_right, estimator_right = build_lane(config["perspective"]["right_lane"], fps, speed_cfg)

    model = YOLO(WEIGHTS_PATH)
    tracker = sv.ByteTrack()
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()

    vehicle_lane = {}  # tracker_id -> "left" | "right", locked on first sighting
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections[[c in VEHICLE_CLASS_IDS for c in detections.class_id]]
        detections = tracker.update_with_detections(detections)

        speeds = []

        if detections.tracker_id is not None:
            for tracker_id, box in zip(detections.tracker_id, detections.xyxy):
                x1, y1, x2, y2 = box
                cx = int((x1 + x2) / 2)
                cy = int(y2)  # bottom-center: where the vehicle touches the road

                if tracker_id not in vehicle_lane:
                    vehicle_lane[tracker_id] = "left" if cx < lane_split_x else "right"

                if vehicle_lane[tracker_id] == "left":
                    warped_point = transformer_left.transform([(cx, cy)])[0]
                    speed = estimator_left.update(tracker_id, warped_point, frame_index)
                else:
                    warped_point = transformer_right.transform([(cx, cy)])[0]
                    speed = estimator_right.update(tracker_id, warped_point, frame_index)

                speeds.append(speed)

                if DEBUG:
                    print(f"Frame {frame_index} | ID {tracker_id} | "
                          f"pixel=({cx},{cy}) -> world=({warped_point[0]:.2f}, {warped_point[1]:.2f}) "
                          f"| speed={speed} km/h")

            annotated = box_annotator.annotate(frame.copy(), detections)
            labels = [
                f"ID {tid} | {model.names[cid]} | {spd:.1f} km/h"
                for tid, cid, spd in zip(detections.tracker_id, detections.class_id, speeds)
            ]
            annotated = label_annotator.annotate(annotated, detections, labels)
        else:
            annotated = frame

        cv2.imshow("Vehicle Speed Monitor", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_index += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
