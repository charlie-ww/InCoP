# Overlap geometry of the high- and low-overlap splits

What the two splits actually are, measured rather than assumed from their names.

## Run it

    python measure_overlap_geometry.py     # writes perframe_overlap.csv
    python read_visibility_subsets.py      # reads the saved evaluation YAMLs

`measure_overlap_geometry.py` reads the dataset annotations directly and needs
no GPU and no checkpoint. `read_visibility_subsets.py` pulls the
`visibility_subset_metrics` block out of the evaluation summaries already saved
under `opencood/logs/isaacsim_ours_pretrained_hospital_2026_07_27_15_59_42/`.

Objects are counted inside the detector's own BEV window
(`[0, -11.2] .. [22.4, 11.2]` in the ego frame), so the counts line up with
what the evaluation scores.

## What it found (hospital test, 3941 frames)

| | low overlap | high overlap |
|---|---|---|
| robot distance, median | 8.04 m | 1.00 m |
| robot distance, p10 / p90 | 6.42 / 14.28 | 1.00 / 1.60 |
| heading difference, median / p90 | 29.2 deg / 127.7 deg | 0.0 deg / 0.2 deg |
| union GT objects | 19953 | 11376 |
| ego_only / shared / partner_only | 45.0 / 32.5 / 22.6 % | 10.2 / 80.6 / 9.1 % |
| shared objects per frame, median | 2 | 6 |
| frames with zero shared objects | 26.0 % | 1.1 % |
| frames with <= 2 shared objects | 53.3 % | 10.9 % |

**The high-overlap split is the same path with the same heading, shifted
1-2.5 m - not two robots in different places.** It is usable as the "redundant
partner" endpoint and should be labelled as such; it does not make overlap a
continuous variable.

*Corrected 2026-11-11.* This section first called it "a 1 m stereo baseline,
partner ~0.9-1.0 m to the side". That came from the test split alone (one
scene, hospital). Across train, validate and test the headings do match
(median difference 0.01-0.08 deg, max 5.3) but the shift is 1.0-2.5 m in
varying directions, with a front-back component up to 0.91 m median in the
warehouse scene, so one robot often appears in the other's camera. See
`runlogs/highoverlap/geometry_all_splits.log`. The table above is still the
test split.

Note the coverage trade-off: the high-overlap split sees 43% fewer distinct
objects (19953 -> 11376), because two robots standing together see the same
things.

## Why the robustness gap is a composition effect

Subset mAP@0.3 retention at `sigma_pos = sigma_rot = 0.8`, each split against
its own noise-free baseline:

| | low overlap | high overlap |
|---|---|---|
| ego_only | 65.2 % | 57.5 % |
| shared | 11.1 % | 54.8 % |
| partner_only | 0.3 % | 0.4 % |

High overlap looks more robust only because 80.6% of its objects are co-visible,
and the ego can detect those by itself. The subset collaboration exists for —
objects only the partner sees — collapses to ~0% in **both** splits. Overall mAP
on the high-overlap split therefore reports health while the collaborative
capability is gone.

## Why the geometry is what it is: by construction

`IsaacSimGenScript/map/planning/planner.py:587`
(`_find_high_overlap_translation`) builds the partner path by translating the
**ego's own path**:

    partner_path = ego_path + shift          # pure translation
    high_overlap_min_dist_m = 1.0            # search lower bound
    high_overlap_max_dist_m = 2.5
    score = (fov_iou, -radius, -angle_deg)   # max FOV overlap, then SMALLEST radius

plus an early break at `first_valid_radius + 0.3`. A pure translation preserves
heading, so the heading difference is exactly 0.00 deg by construction, and the
scoring prefers the smallest feasible shift, which is why the measured distance
sits at p10 = p50 = 1.00 m. `dual.py` labels these cases
`partner_reason: translated_path_with_high_nominal_fov_overlap` and
`ego_reason: inherited_from_<source>_case`.

So the high-overlap "partner" is the ego's own trajectory shifted sideways by
the smallest feasible amount, not an independently planned second robot. The
measurement here and the generator agree.

Two consequences:

- Generating intermediate overlap is a **parameter change**, not new planning
  logic: raise `high_overlap_min_dist_m` and widen the bounds. Rendering still
  needs Isaac Sim.
- `IsaacSimGenScript/main.sh` generates a **fourth** case type,
  `dual_case_random`, for all three scenes, and the trajectory README lists
  `path_case_random/`. It is **not present in the released dataset** - only
  `distance`, `shadow` and `highoverlap` were extracted. Random placement would
  give the continuous overlap axis directly; worth asking the author whether
  that data still exists.

## Caveat that is not yet resolved

The checkpoint was trained on `dataset/IsaacSimOPV2V_hospital/train`, which is
**low overlap only**. Every high-overlap number is therefore confounded by
domain shift, and the noise-free gap (0.3095 vs 0.3987) cannot be attributed to
overlap alone. The high-overlap train split (70 scenes) has never been used for
training.

Verified five ways:

1. `config.yaml.orig` is the author's own file (it still carries his
   `/nfs/home/uuugaga/...` absolute paths) and names the same `root_dir`. The
   only two diffs against `config.yaml` are path fixes; nothing in the data
   section.
2. `opencood/data_utils/datasets/basedataset/opv2v_basedataset.py:48-51` is
   `if self.train: root_dir = params['root_dir'] else: params['validate_dir']`,
   so `root_dir` really is the training split.
3. That directory holds 178 real directories (no symlinks), case types
   `dual_case_distance` and `dual_case_shadow` only, zero `highoverlap`.
4. The encoder it initialises from (`isaac_pretrained`, loading only
   `encoder_m1`, from
   `isaacsim_m6_BEVFusion_center_head_hospital_2026_07_17_07_17_41`) has the
   same `root_dir`, and the string `highoverlap` appears nowhere in its config.
5. The high-overlap data is a separate top-level dataset directory, absent from
   the layout documented in `dataset/README.md`.

**What is not verified.** That checkpoint directory has no training log, no
tensorboard events and no `scripts/` snapshot - only the `.pth` and the two
configs. The claim rests on the recorded config, not on a record of what the
run consumed, and the config stores a *relative* path, so what that directory
held on the author's machine in July 2026 is not knowable from here. Confirming
it properly means asking the author.

## Cross-check

The per-frame measurement here is independent of the evaluation pipeline. It
finds 26.0% of low-overlap frames with no co-visible object; the evaluation
script's own `frames_with_gt` for the `shared` subset implies 26.9%. The two
agree.

## Files

- `measure_overlap_geometry.py` - per-frame measurement from the annotations
- `perframe_overlap.csv` - one row per frame, both splits
- `read_visibility_subsets.py` - subset composition and mAP from the eval YAMLs
- `run.log` - output of the measurement run
