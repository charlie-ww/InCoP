"""Feasibility test for free-space-based alignment uncertainty.

Question: can the contradiction between the ego's observed free space and the
partner's warped returns be used to estimate how badly the relative pose is
misaligned, in a regime where the two robots share almost no field of view?

The statistic is computed exactly as it would be at inference time -- it uses
only the ego point cloud, the partner point cloud and the (noisy) relative
transform. The true alignment error is computed separately, from the clean
poses, and is used only to score the statistic afterwards.
"""

import argparse
import csv
import glob
import os
import sys

import numpy as np
import yaml

sys.path.insert(0, "/nfs/home/charlie/InCoP")
import opencood.utils.pcd_utils as pcd_utils
from opencood.utils.transformation_utils import x1_to_x2

# Detection range of the model, in the ego lidar frame.
LIDAR_RANGE = (0.0, -11.2, -1.0, 22.4, 11.2, 3.0)

# The lidar sits 0.47 m above the floor, so floor returns come back at z = -0.47
# and the ceiling of the hospital scenes at z ~ 2.5. Neither says anything about
# what occupies the robot's own height band, and the floor in particular would
# make every azimuth look blocked at point blank range.
OBSTACLE_Z = (-0.40, 2.00)


def load_frame(agent_dir, timestamp):
    pcd = pcd_utils.pcd_to_np(os.path.join(agent_dir, f"{timestamp}.pcd"))
    with open(os.path.join(agent_dir, f"{timestamp}.yaml")) as handle:
        params = yaml.safe_load(handle)
    return pcd, params["lidar_pose"]


def add_pose_noise(pose, pos_std, rot_std, rng):
    """Same convention as opencood.utils.pose_utils.generate_noise: gaussian on
    x, y and yaw only, with yaw in degrees."""
    noisy = list(pose)
    noisy[0] += rng.normal(0.0, pos_std)
    noisy[1] += rng.normal(0.0, pos_std)
    noisy[4] += rng.normal(0.0, rot_std)
    return noisy


def alignment_error_m(pose_partner, pose_ego, noisy_partner, noisy_ego):
    """Translational part of the residual between the clean and the noisy
    partner->ego transform, matching _set_oracle_gate in the heter model."""
    clean = x1_to_x2(pose_partner, pose_ego)
    noisy = x1_to_x2(noisy_partner, noisy_ego)
    residual = np.linalg.inv(clean) @ noisy
    return float(np.linalg.norm(residual[:2, 3]))


def grid_shape(res):
    x_min, y_min, _, x_max, y_max, _ = LIDAR_RANGE
    return int(round((x_max - x_min) / res)), int(round((y_max - y_min) / res))


def cell_centres(res):
    x_min, y_min, _, x_max, y_max, _ = LIDAR_RANGE
    nx, ny = grid_shape(res)
    xs = x_min + (np.arange(nx) + 0.5) * res
    ys = y_min + (np.arange(ny) + 0.5) * res
    return np.meshgrid(xs, ys, indexing="ij")


def ego_free_space(pcd, res, az_bin_deg):
    """Mark every BEV cell the ego has positively observed to be empty.

    A cell is free when it lies closer to the sensor than the first return along
    its bearing. Cells past the first return are occluded, and bearings with no
    return at all are outside the field of view -- both are unknown, not free,
    and that distinction is the whole point of the experiment.
    """
    band = pcd[(pcd[:, 2] > OBSTACLE_Z[0]) & (pcd[:, 2] < OBSTACLE_Z[1])]
    rng_all = np.linalg.norm(pcd[:, :2], axis=1)
    az_all = np.degrees(np.arctan2(pcd[:, 1], pcd[:, 0]))

    edges = np.arange(-90.0, 90.0 + az_bin_deg, az_bin_deg)
    n_bins = len(edges) - 1

    # First obstacle return per bearing.
    first = np.full(n_bins, np.inf)
    if len(band):
        az_band = np.degrees(np.arctan2(band[:, 1], band[:, 0]))
        rng_band = np.linalg.norm(band[:, :2], axis=1)
        idx = np.clip(np.digitize(az_band, edges) - 1, 0, n_bins - 1)
        np.minimum.at(first, idx, rng_band)

    # Bearings with no obstacle but with floor returns were still swept: the
    # floor was seen out to its furthest return, so that span is observed free.
    reach = np.zeros(n_bins)
    idx_all = np.clip(np.digitize(az_all, edges) - 1, 0, n_bins - 1)
    np.maximum.at(reach, idx_all, rng_all)
    observed = reach > 0.0

    limit = np.where(np.isfinite(first), first, reach)

    gx, gy = cell_centres(res)
    cell_rng = np.hypot(gx, gy)
    cell_az = np.degrees(np.arctan2(gy, gx))
    cell_bin = np.clip(np.digitize(cell_az, edges) - 1, 0, n_bins - 1)

    free = observed[cell_bin] & (cell_rng < limit[cell_bin] - res)
    # The surface the ray stopped at. Free and surface together are everything
    # the ego actually has an opinion about; the rest is behind an obstacle or
    # outside the field of view.
    surface = observed[cell_bin] & (np.abs(cell_rng - limit[cell_bin]) <= res)
    return free, surface


def partner_cells(pcd, transform, res):
    """Partner returns warped into the ego frame, as occupied BEV cells."""
    band = pcd[(pcd[:, 2] > OBSTACLE_Z[0]) & (pcd[:, 2] < OBSTACLE_Z[1])]
    if not len(band):
        return np.zeros(grid_shape(res), dtype=bool), 0
    xyz1 = np.concatenate([band[:, :3], np.ones((len(band), 1))], axis=1)
    warped = (transform @ xyz1.T).T[:, :3]

    x_min, y_min, _, x_max, y_max, _ = LIDAR_RANGE
    inside = (
        (warped[:, 0] >= x_min) & (warped[:, 0] < x_max)
        & (warped[:, 1] >= y_min) & (warped[:, 1] < y_max)
    )
    warped = warped[inside]

    nx, ny = grid_shape(res)
    occupied = np.zeros((nx, ny), dtype=bool)
    if len(warped):
        ix = ((warped[:, 0] - x_min) / res).astype(int)
        iy = ((warped[:, 1] - y_min) / res).astype(int)
        occupied[ix, iy] = True
    return occupied, int(len(warped))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_dir",
                        default="/nfs/home/charlie/InCoP/dataset/IsaacSimOPV2V_hospital/test")
    parser.add_argument("--frames", type=int, default=200,
                        help="number of frames sampled across all cases")
    parser.add_argument("--res", type=float, default=0.2, help="BEV cell size, metres")
    parser.add_argument("--az_bin", type=float, default=0.5, help="bearing bin, degrees")
    parser.add_argument("--ratio", type=float, default=10.0,
                        help="rot_std = ratio * pos_std, the author convention")
    parser.add_argument("--sigmas", default="0.0,0.05,0.10,0.15,0.20,0.30,0.40,0.60,0.80")
    parser.add_argument("--repeats", type=int, default=3,
                        help="noise draws per frame per sigma")
    parser.add_argument("--seed", type=int, default=303)
    parser.add_argument("--out", default="contradiction.csv")
    opt = parser.parse_args()

    sigmas = [float(s) for s in opt.sigmas.split(",")]
    rng = np.random.default_rng(opt.seed)

    cases = sorted(d for d in glob.glob(os.path.join(opt.split_dir, "*")) if os.path.isdir(d))
    frames = []
    for case in cases:
        ego_dir, partner_dir = os.path.join(case, "0"), os.path.join(case, "1")
        if not (os.path.isdir(ego_dir) and os.path.isdir(partner_dir)):
            continue
        for pcd_path in sorted(glob.glob(os.path.join(ego_dir, "*.pcd"))):
            stamp = os.path.basename(pcd_path)[:-4]
            if os.path.exists(os.path.join(partner_dir, f"{stamp}.pcd")):
                frames.append((case, stamp))
    print(f"{len(cases)} cases, {len(frames)} paired frames available", flush=True)

    if len(frames) > opt.frames:
        picks = np.linspace(0, len(frames) - 1, opt.frames).astype(int)
        frames = [frames[i] for i in picks]
    print(f"using {len(frames)} frames x {len(sigmas)} sigmas x {opt.repeats} draws", flush=True)

    with open(opt.out, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["case", "frame", "sigma", "draw", "error_m",
                         "contradiction", "contradiction_obs",
                         "partner_cells", "partner_cells_observed",
                         "ego_free_frac", "ego_observed_frac"])

        for n, (case, stamp) in enumerate(frames):
            ego_pcd, ego_pose = load_frame(os.path.join(case, "0"), stamp)
            par_pcd, par_pose = load_frame(os.path.join(case, "1"), stamp)

            # Depends on the ego alone, so it is computed once per frame.
            free, surface = ego_free_space(ego_pcd, opt.res, opt.az_bin)
            observed = free | surface
            free_frac = float(free.mean())
            observed_frac = float(observed.mean())

            for sigma in sigmas:
                for draw in range(opt.repeats):
                    if sigma == 0.0:
                        noisy_ego, noisy_par = ego_pose, par_pose
                    else:
                        noisy_ego = add_pose_noise(ego_pose, sigma, sigma * opt.ratio, rng)
                        noisy_par = add_pose_noise(par_pose, sigma, sigma * opt.ratio, rng)

                    err = alignment_error_m(par_pose, ego_pose, noisy_par, noisy_ego)
                    transform = x1_to_x2(noisy_par, noisy_ego)
                    occupied, n_pts = partner_cells(par_pcd, transform, opt.res)

                    n_occ = int(occupied.sum())
                    n_obs = int((occupied & observed).sum())
                    hit = int((occupied & free).sum())
                    # Unconditioned: shrinks when the partner drifts out of
                    # range entirely, which is not evidence of agreement.
                    contradiction = hit / n_occ if n_occ else float("nan")
                    # Conditioned on the cells the ego can actually judge.
                    contradiction_obs = hit / n_obs if n_obs else float("nan")

                    writer.writerow([os.path.basename(case), stamp, sigma, draw,
                                     f"{err:.4f}", f"{contradiction:.5f}",
                                     f"{contradiction_obs:.5f}",
                                     n_occ, n_obs,
                                     f"{free_frac:.4f}", f"{observed_frac:.4f}"])
                    if sigma == 0.0:
                        break  # the clean transform is deterministic

            if (n + 1) % 20 == 0:
                print(f"  {n + 1}/{len(frames)} frames", flush=True)

    print(f"wrote {opt.out}", flush=True)


if __name__ == "__main__":
    main()
