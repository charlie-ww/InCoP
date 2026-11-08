"""Estimate how badly two agents are misaligned, from free space alone.

The ego knows more than where it saw something: along every bearing it swept,
everything in front of the first return is space it positively observed to be
empty. When a partner's returns land in that space, the two robots disagree
about the world, and under an indoor complementary pairing that disagreement is
almost always a symptom of pose error rather than of a real object.

This matters because the usual cues run out here. Pose-graph correction needs
objects both agents can see, and in a pairing designed so that the partner
covers what the ego cannot, there often are none. Free space does not need
them: the ego contributes negative evidence, which it has in exactly the places
where it has no positive evidence to contribute.

The statistic is conditioned on the cells the ego can actually judge. Counting
against every partner cell instead lets the score fall as the partner drifts
out of view, which reads as agreement when it is really absence of evidence --
on the hospital test split that choice is the difference between a Spearman
rho of 0.10 and 0.51.
"""

import numpy as np

import opencood.utils.pcd_utils as pcd_utils

# The lidar sits 0.47 m above the floor, so floor returns come back at
# z = -0.47 and the ceilings of the hospital scenes at about 2.5. Neither says
# anything about what occupies the robot's own height band, and the floor in
# particular would make every bearing look blocked at point blank range.
OBSTACLE_Z = (-0.40, 2.00)


def _bearing_bins(az_bin_deg):
    return np.arange(-90.0, 90.0 + az_bin_deg, az_bin_deg)


class FreeSpaceContradiction:
    """Per-sample misalignment score, cached by point-cloud path.

    The ego free-space map and the partner's obstacle band depend only on the
    frame, so they survive across noise levels; only the warp is redone.
    """

    def __init__(self, lidar_range, res=0.2, az_bin=0.5, obstacle_z=OBSTACLE_Z):
        self.lidar_range = tuple(float(v) for v in lidar_range)
        self.res = float(res)
        self.az_bin = float(az_bin)
        self.obstacle_z = tuple(float(v) for v in obstacle_z)
        self._free = {}
        self._band = {}

        x_min, y_min, _, x_max, y_max, _ = self.lidar_range
        self.nx = int(round((x_max - x_min) / self.res))
        self.ny = int(round((y_max - y_min) / self.res))
        xs = x_min + (np.arange(self.nx) + 0.5) * self.res
        ys = y_min + (np.arange(self.ny) + 0.5) * self.res
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        self._cell_range = np.hypot(gx, gy)
        self._cell_az = np.degrees(np.arctan2(gy, gx))

    def _band_points(self, pcd):
        z_lo, z_hi = self.obstacle_z
        return pcd[(pcd[:, 2] > z_lo) & (pcd[:, 2] < z_hi)]

    def _ego_maps(self, pcd):
        """Cells the ego observed to be empty, and the surfaces it stopped at."""
        band = self._band_points(pcd)
        edges = _bearing_bins(self.az_bin)
        n_bins = len(edges) - 1

        first = np.full(n_bins, np.inf)
        if len(band):
            az = np.degrees(np.arctan2(band[:, 1], band[:, 0]))
            rng = np.linalg.norm(band[:, :2], axis=1)
            np.minimum.at(first, np.clip(np.digitize(az, edges) - 1, 0, n_bins - 1), rng)

        # A bearing with only floor returns was still swept, and the floor was
        # seen out to its furthest return, so that span is observed free too.
        reach = np.zeros(n_bins)
        az_all = np.degrees(np.arctan2(pcd[:, 1], pcd[:, 0]))
        rng_all = np.linalg.norm(pcd[:, :2], axis=1)
        np.maximum.at(reach, np.clip(np.digitize(az_all, edges) - 1, 0, n_bins - 1), rng_all)
        observed = reach > 0.0
        limit = np.where(np.isfinite(first), first, reach)

        cell_bin = np.clip(np.digitize(self._cell_az, edges) - 1, 0, n_bins - 1)
        swept = observed[cell_bin]
        free = swept & (self._cell_range < limit[cell_bin] - self.res)
        surface = swept & (np.abs(self._cell_range - limit[cell_bin]) <= self.res)
        return free, surface

    def _cached_ego(self, path):
        if path not in self._free:
            self._free[path] = self._ego_maps(pcd_utils.pcd_to_np(path))
        return self._free[path]

    def _cached_band(self, path):
        if path not in self._band:
            pcd = pcd_utils.pcd_to_np(path)
            self._band[path] = self._band_points(pcd)[:, :3].copy()
        return self._band[path]

    def score(self, ego_pcd_path, partner_pcd_path, transform):
        """Fraction of judgeable partner cells that contradict the ego.

        `transform` is the 4x4 partner-to-ego matrix actually used for fusion,
        so the score reflects the pose the model believes, not the true one.
        Returns (score, n_judgeable); the score is nan when the partner lands
        nowhere the ego can judge, which is itself a fact worth acting on.
        """
        free, surface = self._cached_ego(ego_pcd_path)
        band = self._cached_band(partner_pcd_path)
        if not len(band):
            return float("nan"), 0

        homogeneous = np.concatenate([band, np.ones((len(band), 1))], axis=1)
        warped = (np.asarray(transform, dtype=np.float64) @ homogeneous.T).T[:, :3]

        x_min, y_min, _, x_max, y_max, _ = self.lidar_range
        inside = (
            (warped[:, 0] >= x_min) & (warped[:, 0] < x_max)
            & (warped[:, 1] >= y_min) & (warped[:, 1] < y_max)
        )
        warped = warped[inside]
        if not len(warped):
            return float("nan"), 0

        occupied = np.zeros((self.nx, self.ny), dtype=bool)
        ix = ((warped[:, 0] - x_min) / self.res).astype(int)
        iy = ((warped[:, 1] - y_min) / self.res).astype(int)
        occupied[ix, iy] = True

        judgeable = int((occupied & (free | surface)).sum())
        if judgeable == 0:
            return float("nan"), 0
        return float((occupied & free).sum()) / judgeable, judgeable


def parse_bins(spec):
    """"limit:sigma,..." in ascending limit order, with inf as the last limit."""
    bins = []
    for chunk in spec.split(","):
        limit, value = chunk.split(":")
        bins.append((float(limit), float(value)))
    if any(b <= a for (a, _), (b, _) in zip(bins, bins[1:])):
        raise ValueError(f"bin limits must ascend: {spec}")
    return bins


def sigma_from_score(score, bins, fallback):
    """Pick a blur sigma from the score, falling back when there is no evidence.

    The fallback is not a detail: on the hospital test split the partner lands
    somewhere the ego can judge in about 85% of draws, so the remaining 15% has
    to be handled by a policy rather than by an estimate.
    """
    if not np.isfinite(score):
        return float(fallback)
    for limit, value in bins:
        if score <= limit:
            return float(value)
    return float(bins[-1][1])
