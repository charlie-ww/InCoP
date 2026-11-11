# Upper bound for "just look at the other robot"

The ego's LiDAR hits the partner robot in 52.1% of test frames
(`runlogs/partner_visibility/mutual_sighting.py`). Seeing it gives the partner's
position in the ego's own frame, untouched by either robot's localisation
error. This measures what that is worth, and which part of the pose error -
position or heading - does the damage.

These are upper bounds: the position (or heading) is taken from the true
transform, not from a detector. A real centroid estimate from about 12 points
at 7.5 m would do worse. No blur is applied in these runs.

Sanity check first: under sigma_pos 0.60, substituting the true transform gave
0.4123 against 0.4119 at zero noise on the same 200 samples, so the hook does
what it should.

## mAP@0.3

| sigma_pos | no fix | best blur (constant) | blur oracle | position fixed where seen | position fixed always | heading fixed always |
|---|---|---|---|---|---|---|
| 0.10 | 0.2693 | 0.2810 | 0.2906 | **0.3120** | **0.3511** | - |
| 0.20 | 0.1918 | 0.2506 | 0.2505 | 0.2451 | **0.2961** | 0.2085 |
| 0.30 | 0.1656 | 0.2407 | 0.2419 | 0.2054 | 0.2539 | - |
| 0.60 | 0.1438 | 0.2309 | 0.2304 | 0.1726 | 0.1980 | 0.1461 |
| 0.80 | 0.1427 | 0.2307 | 0.2293 | 0.1628 | 0.1878 | - |

Zero noise: 0.3986.

Share of the pose-error loss recovered:

| sigma_pos | position where seen | position always | heading always |
|---|---|---|---|
| 0.10 | 33.0% | 63.3% | - |
| 0.20 | 25.8% | 50.4% | 8.1% |
| 0.30 | 17.1% | 37.9% | - |
| 0.60 | 11.3% | 21.3% | 0.9% |
| 0.80 | 7.9% | 17.6% | - |

## Findings

1. **At small error it is the strongest thing measured so far.** At
   sigma_pos 0.10, fixing position only where the robot is actually seen
   (0.3120) beats the best blur (0.2810) and even the blur oracle (0.2906).
   Any thesis method has to be compared with it.
2. **At larger error, position alone is not enough and unhedged it loses to
   blurring.** From 0.20 up, "position where seen" is below the best blur, and
   even "position always" is below it from 0.60. The leftover heading error is
   enough to break small objects (the heading budget is about 1.15 deg at 8 m)
   and nothing hedges it here.
3. **What the split separates is "where the partner is" from "which way its
   map is turned" - not translation noise from rotation noise.** Fixing the
   partner's position recovers 50.4% at 0.20; fixing only the relative heading
   recovers 8.1%. That does NOT mean rotation noise is minor. Noise is on both
   robots, and the ego's own heading error swings the partner's believed
   position around the ego (2 deg at 8 m is 0.28 m). That part lands in the
   position column, so seeing the partner removes it and "heading fixed" does
   not. Independent check from `figures/make_overlap_ratio_figure.py`: cutting
   the rotation noise from 10x to 1x the translation noise recovers 33% of the
   loss at 0.10 and 19% at 0.20 - more than "heading fixed" (8.1%). Corrected
   2026-11-12; the first version of this README said position error was the
   larger share, which mixed the two.
4. **The benefit scales with how often the robot is seen.** "Where seen"
   recovers 45-53% of what "always" does, close to the 52.1% of frames in which
   it is seen. The frames with sightings are about average, not easy ones.

## Caveats

- Noise goes on both robots' poses, sigma_pos metres on x and y and 10 x
  sigma_pos degrees on yaw, so the relative heading error is about 1.4 x 10 x
  sigma_pos degrees (2.8 deg at 0.20). That 10:1 ratio is this project's
  convention; outdoor papers often use equal numbers (0.6 m with 0.6 deg).
  With smaller heading noise, fixing position would recover more. The
  strength of this baseline depends on that ratio, which should be justified
  or swept.
- No blur on top. "Fix position where seen, blur the rest" is the obvious
  combination and was not run.
- Oracle position, not a detector.

## Fix the position where seen, then blur (direct_obs_blur_sweep.sh)

Added 2026-11-12. Same upper bound for the position, plus the hedge the first
sweep lacked: frames where the partner is seen get blur sigma S for the heading
error left over; frames where it is not seen get the best constant blur
(sigma 2). Sanity first: the per-sample blur hook under sigma_pos 0.60 with the
true transform and blur 2 gave 0.3283, identical to a plain fixed blur of 2 at
zero noise on the same 200 samples.

mAP@0.3:

| sigma_pos | best blur alone | position where seen, no blur | + blur S=0 | + blur S=1 | + blur S=2 |
|---|---|---|---|---|---|
| 0.10 | 0.2810 | **0.3120** | 0.3034 | 0.2904 | 0.2713 |
| 0.20 | 0.2506 | 0.2451 | - | **0.2691** | - |
| 0.30 | 0.2407 | 0.2054 | - | **0.2550** | - |
| 0.60 | 0.2309 | 0.1726 | 0.2036 | 0.2321 | **0.2420** |
| 0.80 | **0.2307** | 0.1628 | - | 0.2279 | - |

"+ blur S" means sigma S where seen and sigma 2 where not; "position where seen,
no blur" is pos_visible from the first sweep (no blur anywhere).

1. **The combination beats blur alone from 0.10 to 0.60**, by +0.031 (no blur
   at all), +0.019, +0.014 and +0.011. At 0.80 the one setting tried (S=1) is
   0.003 below; S=2 was not run there.
2. **The right amount of blur grows with the error.** At 0.10 any blur hurts,
   even on unseen frames (0.3120 with none against 0.3034 with sigma 2 on
   unseen frames). At 0.60 the most blur tried is best (S=2, 0.2420). This is
   the same pattern as the blur sweep without correction, where the best
   constant went from none to sigma 2 as the noise grew - so the blur strength
   should be set by the estimated noise regime, which the regime selector
   already estimates.
3. Over the five levels, the best tried setting per level averages 0.2612,
   against 0.2468 for the best blur per level and 0.2485 for the per-frame blur
   oracle: correcting where possible and hedging the rest beats any amount of
   cleverness about the blur alone.

Caveats: the position is oracle; S was picked on the test split, so the best
per level is a hindsight upper bound; S = 0 and S = 2 were run only at 0.10 and
0.60.
