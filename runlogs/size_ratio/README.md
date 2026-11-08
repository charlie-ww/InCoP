# Object size versus the misalignment it has to survive

Outdoors a 0.3 m pose error slides a car by a fifteenth of its own length and
the box still overlaps itself. Indoors the objects are small enough that the
same error moves them clear of themselves. This is the cheapest argument for
why the dominant outdoor fix cannot transfer.

    python measure_size_ratio.py      # no GPU, no checkpoint, reads annotations

`displacement that breaks IoU@t` is found numerically: the box is shifted
against itself in 72 directions and the mean IoU is bisected until it falls
below t. Sizes are medians of `2 * extent` from the test-split annotations.

## Result

| class | BEV size | breaks IoU@0.3 |
|---|---|---|
| potted_plant | 0.90 m | 0.43 m |
| medical_bag | 0.67 m | 0.32 m |
| chair | 0.66 m | 0.32 m |
| traffic_cone | 0.33 m | 0.16 m |
| trash_can | 0.25 m | 0.12 m |
| wet_floor_sign | 0.25 m | 0.12 m |
| fire_extinguisher | 0.23 m | **0.11 m** (one BEV cell is 0.10 m) |
| **median class** | 0.33 m | **0.16 m** |
| outdoor reference: 1.8 x 4.5 m car | 2.84 m | 1.23 m |

The project's sigma_pos = 0.2 setting produces an actual misalignment of 0.43 m
near the robot and 1.26 m at the edge of the sensing range, so:

- 0.43 m is **267%** of the median object's breaking distance; 6 of 7 classes
  are already broken.
- 1.26 m is **782%**; all 7 are broken.
- The standard outdoor benchmark noise of 0.2 / 0.5 m is **16% / 41%** of a
  car's breaking distance. Both are safe.

A useful coincidence: 1.23 m (where a car starts to break) is almost exactly
1.26 m (the indoor misalignment at range). The same magnitude of error merely
begins to threaten a car and completely destroys every indoor class.

## Why coarsening cannot rescue it

Tolerating a 1.26 m shift means blurring to roughly that scale. The median
indoor object is 0.33 m across - 0.27x the blur scale, so it is erased. A car
is 5.7x the 0.5 m scale an outdoor method needs, so it survives. That is the
structural reason the multi-scale route (CoAlign, MRCNet) does not transfer
indoors, and it is a property of object size, not of the method.

## Caveats

- Boxes are treated as axis-aligned for the IoU sweep. The dataset's
  annotations carry `angle: [0, 0, 0]` for these objects, so this costs little
  here, but a rotated-box IoU would be the rigorous version.
- The car reference is a nominal 1.8 x 4.5 m and a quoted benchmark noise
  range, not a measurement from an outdoor dataset.
