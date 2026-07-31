# Traffic Speed Monitor

YOLO detection + ByteTrack tracking + per-lane homography, to estimate
vehicle speed (km/h) from a fixed highway camera with two carriageways.

## Structure

```
traffic-speed-monitor/
├── app.py                    # main pipeline: run this to watch it live
├── calibrate_perspective.py  # step 1: click road quad per lane
├── calibrate_scale.py        # step 2: click a car's length per lane -> real-world scale
├── config.yaml                # all settings + calibration output live here
├── requirements.txt
├── utils/
│   ├── perspective.py        # PerspectiveTransformer (pixel <-> bird's-eye)
│   └── speed.py               # SpeedEstimator (bird's-eye movement -> smoothed km/h)
├── videos/                    # put your source video here
└── weights/                   # put yolo11s.pt (or your model) here
```

## Setup

1. `pip install -r requirements.txt`
2. Put your video in `videos/` and your YOLO weights in `weights/`.
3. Edit `config.yaml` -> `video.path` to match your video's filename.

## Calibration (do this once per camera angle)

**Step 1 - road quad (perspective):**
```
python calibrate_perspective.py
```
Click 4 points per lane (top-left, top-right, bottom-right, bottom-left),
following the road. A bird's-eye preview pops up after each lane so you
can sanity-check it before moving on -- lane markings and road edges
should look roughly vertical and parallel, and cars shouldn't look
absurdly stretched. This writes `source_points` into `config.yaml`.

**Step 2 - real-world scale:**
```
python calibrate_scale.py --lane left_lane
python calibrate_scale.py --lane right_lane
```
Click front-then-rear of a car (or any two points a known distance
apart) on a few different vehicles. This writes `meters_per_unit`
into `config.yaml` for that lane. Run it once per lane -- the two
carriageways almost never have the same scale, since their road quads
cover different physical distances.

## Run it

```
python app.py
```

`app.py` will refuse to start if either lane is missing
`source_points` or `meters_per_unit` -- run the two calibration steps
first.

## Tuning

All speed-smoothing knobs live under `speed_estimation` in
`config.yaml`:
- `ema_alpha` -- higher = speed reading reacts faster to real changes, but is jumpier
- `min_distance_m` -- movement below this is treated as detector jitter, not real motion
- `max_plausible_kmph` -- single-frame readings above this are rejected as noise/ID-switches
- `max_frame_gap` -- if a track was lost longer than this many frames, don't trust the jump when it reappears

## Notes

- `output_width` / `output_height` in `config.yaml` are just the
  warp canvas resolution, not real-world units -- don't touch these
  to "fix" scale, that's what `meters_per_unit` is for.
- Set `DEBUG = True` at the top of `app.py` to print per-vehicle
  pixel/world coordinates and speed each frame.
