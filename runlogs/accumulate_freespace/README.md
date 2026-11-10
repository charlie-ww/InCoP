# Pilot: if the pose error is constant over an episode, does evidence add up?

**Motivation.** The benchmark draws the pose error fresh every frame (0.2 m at
the project's default), while the robots move 0.08 m per frame. A real
localisation error cannot change by more than the robot moves; it is close to
constant over an episode. That i.i.d. convention comes from the outdoor
benchmarks. If the error is constant, each frame is evidence about one unknown.

**Test.** Truth is offset 0. A 13 x 13 grid of world-frame translations
(+-0.6 m, 0.1 m steps, yaw held at truth) is scored on every frame with the
free-space contradiction, pooled over N frames, and the lowest-cost candidate
graded against the truth. CPU only, 163 s.

## Results (regrade.py - see the note on ties)

All windows, all episodes:

| frames pooled | median error | within 0.16 m | within 0.30 m |
|---|---|---|---|
| chance | 0.496 m | 5.3% | - |
| 1 | 0.374 m | 21.8% | 40.4% |
| 10 | 0.305 m | 28.6% | 49.7% |
| 30 | 0.283 m | 36.8% | 50.9% |

Pooling helps, slowly. Consecutive frames are 8 cm apart and see nearly the
same thing, so thirty frames are far from thirty independent looks.

## Two findings that matter more than the headline (surface_width.py)

**It fails exactly where there are no co-visible objects.**

| co-visible objects | frames | median error | within 0.16 m |
|---|---|---|---|
| 0 | 432 | 0.442 m | **1.2%** |
| 1-2 | 513 | 0.457 m | 15.4% |
| 3-5 | 724 | 0.316 m | 27.8% |
| >=6 | 566 | 0.224 m | 35.7% |

At zero co-visible objects the one correspondence-free geometric cue we have is
at chance (1.2% against 5.3%). This is the measurement behind the claim that
those frames carry no information to correct the pose with.

**The width of the low-cost region is a weak uncertainty.** Spearman +0.151
with the error. The widest third is reliably bad (6.2% within 0.16 m) but the
narrowest third is right only 32.8% of the time, and inside the zero-co-visible
frames the relation reverses (-0.251). It can say "do not trust this"; it
cannot say "trust this".

## Caveats

- Optimistic setup: translation searched with yaw known, truth inside the grid.
  A joint search would do worse.
- pilot.py averaged tied minima, which pulls a flat surface to the grid centre,
  where the truth is. regrade.py grades a tie by the mean distance of the tied
  candidates instead. Ties turned out rare (median 1) but the numbers above are
  the regraded ones; the table printed by pilot.py is superseded.
- Frames from the start of an episode score better than the average frame
  (39% against 21.8% within 0.16 m), so "first N frames" tables flatter it.
