# Can each robot read its heading off the building, with no shared objects?

**Why.** Seeing the partner robot fixes its position (`runlogs/direct_obs`), but
not its heading, and heading is what breaks small objects at range: the
tolerance for a median 0.33 m object at IoU 0.3 is about 1.15 deg at 8 m and
0.41 deg at 22.4 m. Same-frame shared objects, the field's usual heading
evidence, are what low overlap removes.

**Idea.** Indoor buildings are built on two perpendicular directions. If each
robot measures, from its own scan, the angle alpha of those directions in its
own frame, then alpha_ego - alpha_partner = yaw_partner - yaw_ego (mod 90 deg).
The two robots need to see nothing in common, only the same building. The
90 deg ambiguity is resolved by the reported heading, whose error is far below
45 deg.

**Method** (`measure_heading.py`). Wall band of the scan (sensor z -0.30 to
2.25 m), 5 cm xy cells, rotate by each candidate angle and score how sharply the
cells pile up when projected on the two axes; 0.5 deg grid, 0.05 deg
refinement. Chosen over a local-line PCA variant on 8 validation cases
(`compare_variants.log`: median 0.50 vs 0.98 deg). Truth is the GT poses; the
estimate itself never uses a pose. CPU only, about 1 minute for 11,231 frames.

Tuning (cut points, calibrated ranges) is on warehouse and office (train +
validate); hospital test is a building the method has never seen.

## 1. The simulated buildings are axis-aligned

World angle of the walls: warehouse 89.97, hospital 89.95, office 90.00 deg
(mod 90). Single-robot estimate vs that axis, hospital: median 0.38 deg,
76.3% within 1.15 deg.

## 2. Relative heading from one frame

| split | scene | median | <0.41 deg | <1.15 deg | <2 deg | >5 deg |
|---|---|---|---|---|---|---|
| test | hospital | 0.71 | 33.5% | 62.3% | 76.6% | 8.8% |
| train | warehouse | 0.38 | 52.3% | 78.8% | 88.5% | 1.5% |
| train | office | 0.84 | 31.4% | 59.7% | 74.4% | 8.0% |

It works where same-frame evidence is absent (hospital):

| objects both robots see | frames | median | <1.15 deg | >5 deg |
|---|---|---|---|---|
| 0 | 620 | 0.84 | 56.8% | 11.5% |
| 1-2 | 525 | 0.70 | 59.2% | 9.3% |
| 3-5 | 728 | 0.69 | 63.2% | 11.0% |
| >=6 | 567 | 0.61 | 70.2% | 2.5% |

Against the injected heading noise (hospital, within 1.15 deg of the true
relative heading, 20 noise draws per frame):

| sigma_pos | noise std on relative heading | reported | walls (one frame) |
|---|---|---|---|
| 0.1 | 1.41 deg | 58.4% | 62.3% |
| 0.2 | 2.83 deg | 31.6% | 62.3% |
| 0.3 | 4.24 deg | 21.7% | 62.3% |
| 0.6 | 8.49 deg | 10.8% | 62.3% |
| 0.8 | 11.31 deg | 8.3% | 62.3% |

The wall estimate does not depend on the noise; the 90 deg ambiguity is
resolved wrongly in at most 0.012% of draws (sigma 0.8).

**But a single scan barely knows when it is wrong.** Its confidence (peak over
best rival angle) correlates with the error at only -0.16 (hospital). A 90%
range calibrated per confidence tercile on warehouse/office covers 84.7% on
hospital. Gating on that confidence did not help: the gate chosen on
warehouse/office was "always use the walls".

## 3. Pooling each robot's own past frames fixes both problems

Each robot takes the circular median of its last k wall angles, converted with
how much it has itself turned since (here from the true yaw increments; a gyro
or odometry is accurate over the ~1.6 m of a 20-frame window). In order, past
frames only.

| k | median | <0.41 deg | <1.15 deg | <2 deg | >5 deg |
|---|---|---|---|---|---|
| 1 | 0.71 | 33.5% | 62.3% | 76.6% | 8.8% |
| 5 | 0.65 | 37.0% | 64.1% | 77.5% | 7.9% |
| 10 | 0.56 | 40.2% | 66.8% | 79.7% | 5.3% |
| 20 | 0.48 | 44.8% | **76.5%** | 89.6% | **0.6%** |

And the disagreement inside the window is an uncertainty: at k = 10 its
Spearman with the error is **+0.636** on hospital (+0.666 on
warehouse/office). A 90% range calibrated per spread tercile on
warehouse/office has half-widths 0.69 / 1.27 / 4.62 deg and covers 87.4% on
hospital (89.3 / 86.6 / 87.1% per tercile), mean half-width 2.72 deg. The 90%
range of the injected noise itself is 2.33 deg at sigma 0.1 and 4.65 deg at 0.2.

Rejecting a wall reading that disagrees with the reported heading by more than
3 noise std adds a little at sigma 0.1 (66.8% -> 70.8% within 1.15 deg at k = 10)
and nothing above.

## 4. Real robots (`real_world.py`, partial)

The LiDAR robot's wall angle stays put in world coordinates while it moves: in
the train recording 93.0% of frames within 1.15 deg of its own median (it
turned 16.8 deg), in the validate recording 32.8% (median 1.60 deg, it turned
3.4 deg). The second robot's points are layered at fixed forward distances
(about 700 distinct x values in 10k points, like a depth camera); its estimate
sticks at 0 deg in the sensor frame, and spreading points within the layer gaps
(`--dequantize`) does not rescue it. Relative heading on the real pair is
therefore not evaluable with this estimator.

## What it means

- Under low overlap the heading is not unobservable: the building itself is
  shared. A one-frame wall estimate already beats the reported heading at
  every noise level from 0.2 up, including frames with zero shared objects.
- The uncertainty comes from time, not from the single scan: the spread of a
  robot's own recent wall readings ranks the error well and gives a 90% range
  that nearly transfers to an unseen building (87.4%).
- Combined with the partner sighting (position, 52.1% of frames), this covers
  both halves of the alignment without any shared object.

## Caveats

- Simulated walls are perfectly straight and the LiDAR noise-free; the buildings
  are exactly Manhattan. Real buildings have angled walls, glass and clutter.
  The one real LiDAR is encouraging, not a test of relative heading.
- Pooling uses the true yaw increments as the robot's own turning; under the
  benchmark's per-frame i.i.d. pose noise the reported increments would be
  useless, which is another reason the noise model needs settling.
- Calibration transfer is 87.4%, not 90%: a new building shifts the error
  distribution a little.
- Failures (>5 deg) are not yet diagnosed; robots 0-4 m apart do badly (40%, but
  only 25 frames).

## Files

`measure_heading.py` -> `heading.csv` (`run_measure.log`); `analyse.py` ->
`analysis.txt`; `compare_variants.py` -> `compare_variants.log`;
`real_world.py` -> `real_world_raw.log`, `real_world_dequantized.log`.
