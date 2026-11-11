# Can the ego see the partner robot itself?

The project's central tension is that the partner is useful precisely because it
looks where the ego cannot, which is also why there is no shared observation to
align with. Seeing the *partner robot* would break that tension for free: it is
one correspondence that needs no scene overlap at all. It is also the strongest
objection to this whole research direction, so it is worth knowing its size.

Test split, 2,440 frames, 89 degree forward wedge, BEV crop x in [0, 22.4],
y in [-11.2, 11.2].

## What section 1 actually measures - read this first

Section 1 is **not** a measurement of the ego seeing the partner. It is pure
geometry: the partner's **ground-truth position** is read from the annotations,
transformed into the ego frame, and tested for membership of the sensing wedge,
and then a straight line to it is walked through the ego's own obstacle
returns. The correct reading is "the partner's position lies inside the region
the ego's sensor covers, with nothing in the way".

Two consequences, which must always be stated with the number:

- **It is counterfactual.** It says what the ego *would* receive if the robot
  were rendered. It is therefore an upper bound on what direct observation
  could offer, not a count of frames where it works today, which is zero
  (section 2).
- **It uses information the ego does not have.** The test keys off the
  partner's true pose, and not knowing that pose is the problem this project
  exists to study.

So section 1 answers "how often does the trajectory geometry permit direct
observation, seen from outside the system". It bounds an opportunity; it is
not an observation. The phrase "sees the partner" is avoided below for that
reason.

## 1. The geometry is very favourable

| | frames | share |
|---|---|---|
| ego's wedge contains the partner's position | 1,804 | **73.9%** |
| partner's wedge contains the ego's position | 373 | 15.3% |
| at least one direction | 1,971 | 80.8% |
| both directions | 206 | 8.4% |

Median inter-robot distance 8.04 m; when in the wedge, median bearing 13.9 deg
off boresight. Line of sight clear of the ego's own obstacle returns in 70.0%
of in-wedge frames.

The asymmetry (73.9% vs 15.3%) is a property of the generator: the partner
trails or flanks the ego rather than facing it.

## 2. CORRECTED 2026-11-11: the partner robot IS in the LiDAR

The first version of this section said the robot was not rendered. That was
wrong. The camera images show the other robot, and so does the LiDAR
(`recheck_robot_lidar.py`):

| returns within 0.45 m, more than 7 cm above the floor | median | none at all |
|---|---|---|
| low-overlap test, at the other robot (median 7.5 m away) | **12** | 2.0% |
| low-overlap test, at a control position | 0 | 84.0% |
| high-overlap train, at the other robot (median 1.4 m away) | **568** | 0.0% |
| high-overlap train, at a control position | 0 | 57.2% |

What went wrong: the robot is low. Its body sits between z = -0.45 and 0 in the
sensor frame, the top being about level with the LiDAR. The original test took
everything below z = -0.20 to be floor and discarded most of the body. "The ego
sees the floor at the partner's position" was not evidence either: a 0.5 m
circle is larger than the robot, so there is always floor around it.

The "none at all" column comes from a sample covering only the first ten or so
episodes. Over the whole test split (`mutual_sighting.py`, >= 3 returns on the
body counts as seen): the ego's LiDAR hits the partner in 52.1% of frames,
about seven in ten of the 73.9% where it is inside the wedge, which matches the
30% of lines of sight found blocked above. The partner hits the ego in only
3.6% (the generator puts the partner ahead of the ego, facing the same way),
at least one sees the other in 55.2%, and both see each other in 0.6%. Only
mutual sighting fixes the relative heading as well as the position. Direct observation of the partner can be tested
on this dataset as it stands. The robot is still not an annotated class.

Sections 1, 3 and 4 are unaffected: they are geometry and arithmetic, and the
heading-budget argument in section 4 now matters more, not less, because the
baseline it argues against is runnable.

## 3. It fails in the same direction as everything else, but only partly

Having the partner inside the wedge is positively correlated with scene co-visibility
(Pearson +0.304, Spearman +0.315), so it shares the free-space signal's
weakness: most available when least needed. But the correlation is weak enough
to leave real coverage of the hard frames:

| co-visible scene objects | frames | partner's position inside the wedge |
|---|---|---|
| 0 | 634 | **51.7%** |
| 1 | 329 | 78.7% |
| 2 | 337 | 70.3% |
| 3-5 | 756 | 74.9-84.0% |
| >= 6 | 384 | 98-100% |

**In 51.7% of the frames with zero co-visible objects the partner's position is
still inside the ego's wedge.** Those are exactly the frames that defeat CoAlign, HERO and
V2X-PC. Direct observation would cover half of them.

Where it fails, it fails because the robots face away from each other: median
heading difference 45.6 deg when the partner is not visible against 24.2 deg
when it is.

## 4. Why it still does not rescue the problem

Detecting the partner would give you its **position**. Position is not what breaks us.

A partner heading error of delta displaces an object it reports at range r from
itself by r * delta. With the median indoor object breaking IoU@0.3 at 0.161 m:

| object's range from the partner | indoor heading budget | outdoor car budget |
|---|---|---|
| 4 m | 2.31 deg | 17.62 deg |
| 8 m | **1.15 deg** | 8.81 deg |
| 15 m | 0.62 deg | 4.70 deg |
| 22.4 m | **0.41 deg** | 3.15 deg |

So even with the partner's position known exactly, its **yaw** has to be right
to about one degree for mid-range objects and 0.4 degrees at the edge of its
range, against 8.8 and 3.1 degrees for an outdoor car. At 8 m a robot is a
small, roughly symmetric blob; recovering its yaw to a degree from one LiDAR
view is not a solved problem.

This is the honest answer to "why not just look at the other robot", and it is
the same order-of-magnitude argument as the object-size result: the indoor
tolerance is about 8x tighter, so a method that is sufficient outdoors is not
sufficient here.

## Files

- `measure_partner_visibility.py` / `run_visibility.log` - tests 1-3
- `cross_overlap.py` / `run_cross.log` - sections 3 and 4
- `partner_visibility_test.csv` - per-frame, joinable with
  `runlogs/overlap_geometry/perframe_overlap.csv` on (case, frame)
