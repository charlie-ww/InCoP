# Is per-frame pose uncertainty worth estimating at all?

The sweep behind this directory asks one question: given that the ego cannot see
its own pose error, how much accuracy can it recover by *estimating* that error
and blurring the partner's features accordingly?

Six injected noise levels, the test split of `IsaacSimOPV2V_hospital`, the
low-overlap checkpoint `isaacsim_ours_pretrained_hospital_2026_07_27_15_59_42`.
`sigma` below is the Gaussian blur applied to the warped partner feature map;
`sigma_pos` is the injected pose noise in metres (rotation noise is 10x it, in
degrees).

## mAP@0.3

| variant | 0.00 | 0.10 | 0.20 | 0.30 | 0.60 | 0.80 | mean |
|---|---|---|---|---|---|---|---|
| no blur | 0.3986 | 0.2693 | 0.1918 | 0.1656 | 0.1438 | 0.1427 | 0.2186 |
| fixed sigma=1 | 0.3411 | 0.2810 | 0.2368 | 0.2131 | 0.1987 | 0.1949 | 0.2443 |
| fixed sigma=2 | 0.2805 | 0.2665 | **0.2506** | 0.2407 | **0.2309** | **0.2307** | 0.2500 |
| estimated (one calibration) | 0.3422 | 0.2791 | 0.2436 | 0.2337 | 0.2245 | 0.2221 | 0.2575 |
| estimated (regime-adaptive) | 0.3655 | 0.2751 | 0.2402 | 0.2311 | 0.2219 | 0.2223 | **0.2594** |
| oracle (true error per frame) | 0.3984 | 0.2906 | 0.2505 | 0.2419 | 0.2304 | 0.2293 | 0.2735 |
| **best constant for the level** | 0.3986 | 0.2810 | 0.2506 | 0.2407 | 0.2309 | 0.2307 | **0.2721** |

The last row uses no per-frame information whatsoever: it picks one of the three
constants and applies it to the whole level.

## The result

**Per-frame estimation is worth almost nothing here.** Measured against the best
single constant (fixed sigma=2, 0.2500):

| decision | mean mAP@0.3 | gain | share of the oracle gain |
|---|---|---|---|
| best single constant | 0.2500 | +0.0000 | 0% |
| per-frame estimated sigma | 0.2575 | +0.0076 | 32.1% |
| regime-adaptive calibration | 0.2594 | +0.0094 | 39.8% |
| **perfect per-LEVEL constant** | **0.2721** | **+0.0221** | **93.9%** |
| perfect per-FRAME oracle | 0.2735 | +0.0235 | 100% |

Knowing the true error of *every individual frame*, rather than just which
regime the run is in, is worth **+0.0014 — 6.1% of the total gain.** The other
93.9% is available from a single global decision.

Sharper still: **at every sigma_pos >= 0.20 the oracle does not beat the
constant** (-0.0001, +0.0012, -0.0005, -0.0014). The whole oracle advantage sits
at sigma_pos = 0.00 (+0.1179) and 0.10 (+0.0241). Once the pose is genuinely
bad, perfect knowledge of how bad buys nothing over always blurring.

This is a negative result about an action space, not about an estimator. The
free-space score correlates with the true error at rho = 0.506 and the
perturbation probe with per-frame damage at rho = 0.424; both are real signals.
There is simply almost nothing for a per-frame signal to buy when the only
available action is how much to blur.

It also cuts across the field: CoAlign, V2VLoc and UECP all estimate and act
per frame. On this benchmark that granularity is 6% of the reachable gain.

## The regime-adaptive run

`regime_adaptive.sh` picks the calibration from the median of the previous 20
frames' scores (threshold 0.55) and then maps the current frame's score through
it. The regime is decided from earlier frames only, so no frame influences its
own calibration.

It is the best deployable variant (0.2594), but the gain is not free: **it buys
+0.0233 at sigma_pos = 0.00 and pays about -0.003 at each of the four middle
levels.** It is also still below fixed sigma=2 at every sigma_pos >= 0.20. If a
deployment is mostly misaligned, the constant remains the right answer.

Firing rate, deployed: 71.4 / 40.7 / 20.2 / 8.0 / 3.0 / 0.0 % of samples chose
the well-localised calibration, monotone in the injected noise. The detector
works as a sensor; the cap is on what the sensor's output can be spent on.

## Correction: the threshold bootstrap was optimistic

The threshold 0.55 was chosen from `runlogs/freespace/contradiction.csv` by
resampling 20 scores **i.i.d.** within a level, which predicted 92.7%
recognition at sigma_pos = 0 and 1.8% false alarms at 0.20. Deployment gave
71.4% and 20.2%.

The bootstrap was wrong, not the code. A deployed window is 20 *consecutive*
frames, so it almost always lies inside one case and inherits that case's
offset instead of averaging over all 63. Re-running the same csv with a rolling
in-order window reproduces deployment closely:

| sigma_pos | i.i.d. bootstrap | rolling in order | deployed |
|---|---|---|---|
| 0.00 | 92.7% | 77.0% | 71.4% |
| 0.10 | 26.8% | 38.9% | 40.7% |
| 0.20 | 1.8% | 15.6% | 20.2% |
| 0.30 | 0.2% | 7.1% | 8.0% |

The earlier figures of 91.4% / 5.3% quoted elsewhere are the i.i.d. ones and
should be read as 77% / 16%. This is the second time in this project that
i.i.d. resampling overstated separability by averaging away between-case
structure, and it is the same structure that makes evidence poverty an episode
property rather than a frame property.

## Files

- `analyse_regime.py` / `analysis_regime.txt` - the tables above
- `analyse_regime_firing.py` / `analysis_regime_firing.txt` - the correction
- `{none,fixed1,fixed2,estimated,est_regime,oracle}_<sigma_pos>.log` - raw runs
- `est_{fb0,hi,both}_<sigma_pos>.log` - the 18-run recalibration grid, all of
  which came in below the original single calibration (0.2514 / 0.2542 /
  0.2491 against 0.2575)
