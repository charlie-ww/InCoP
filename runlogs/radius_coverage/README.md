# Is an estimated radius better than a fixed one for objects only the partner sees?

The proposal is that in the ego's blind region the partner reports "an object
about here, give or take r" instead of a confident position. This tests the one
thing that proposal depends on: does estimating r buy anything over using the
same r rule for every report?

No detector, no GPU (39 s). Reports are the annotated objects only the partner
sees, inside the ego's BEV window: 28,152 reports over six noise levels. A pose
error on the partner (sigma_pos on x and y, 10 x sigma_pos degrees on yaw,
redrawn per frame) moves each report. Every method is conformally calibrated to
90% coverage on the even-numbered episodes and graded on the odd ones, so the
methods differ only in how the radius is chosen.

## Result

| radius chosen from | coverage | mean radius | reports with r <= 0.5 m |
|---|---|---|---|
| fixed rule, no estimate | 90.1% | 1.54 m | 0.0% |
| this frame's free-space score | 91.2% | 1.43 m | 2.4% |
| median score of previous 20 frames | 91.4% | **1.35 m** | 12.6% |
| true noise level (oracle) | 92.0% | 1.08 m | 34.6% |
| true displacement per object (bound) | 100% | 0.60 m | 59.1% |

Same honesty, smaller circles: 12% smaller with the window score, 30% if the
noise level were known exactly.

The mean hides the more important difference, which is whether "90%" is true
when it matters. Coverage by true noise level:

| | 0.0 | 0.1 | 0.2 | 0.3 | 0.6 | 0.8 |
|---|---|---|---|---|---|---|
| fixed | 100% | 100% | 100% | 99.6% | 79.6% | **61.3%** |
| frame score | 100% | 100% | 99.3% | 96.6% | 80.9% | 70.6% |
| window score | 100% | 99.5% | 93.8% | 90.6% | 86.5% | 78.1% |
| oracle level | 100% | 88.7% | 91.0% | 90.4% | 90.0% | 91.7% |

A fixed radius is 90% on average by being wastefully large when the pose is
fine and wrong four times in ten when it is not. The window score moves the
radius from 0.70 m to 2.03 m across the levels and stays much closer to its
promise.

## What it says about the design

- The window beats the single frame again, and in the frames with zero
  co-visible objects the single-frame score gives nothing (1.52 m against
  1.51 m fixed) while the window still helps (1.37 m). Evidence from other
  frames is the only evidence those frames have.
- The remaining gap is to knowing the regime (1.35 m against 1.08 m), not to
  per-object knowledge the ego could plausibly get. Better regime estimation
  is where the next gain is.
- The circles are large. With the noise mix going up to 0.8 m and 8 degrees, a
  90% circle averages over a metre for objects 0.33 m across. At sigma_pos of
  0.1 to 0.2 the oracle-level radius is 0.33 to 0.67 m. Whether a circle that
  size is useful depends on what consumes it, which this does not measure.

## Caveats

- The partner's reports are ground-truth objects: an ideal detector. Detector
  error would add to every radius.
- Noise is on the partner's pose only and independent per frame, like the
  benchmark runs. Under an error that persists through an episode the window
  would be estimating the actual offset rather than its typical size.
- The growth of displacement with range uses the known 10 degrees per metre
  ratio between rotation and translation noise; only the scale is calibrated.
- Five score bins plus one for "no evidence"; not tuned.
