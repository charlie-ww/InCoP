"""How much does the detection output move when the partner pose is jiggled?

The ego cannot see how wrong its belief about the partner is, but it can ask a
question that needs no shared evidence at all: if the partner were somewhere
else within my localisation prior, would I report anything different? Frames
whose answer is "no" are safe whatever the true error happens to be; frames
whose answer is "a lot" are the ones worth hedging on.

Nothing here reads ground truth. The perturbations come from the prior the ego
would already have, and the statistic is computed from the model's own outputs.
"""
import math

import numpy as np
import torch


def draw_perturbations(k, pos_std, rot_std_deg, rng, include_identity=True):
    """k offsets drawn from the localisation prior, as (dx, dy, dyaw_deg).

    The first draw is the identity when include_identity is set, so the caller
    gets the unperturbed run as the reference without paying for an extra pass.
    """
    out = []
    if include_identity:
        out.append((0.0, 0.0, 0.0))
    while len(out) < k:
        out.append((
            float(rng.normal(0.0, pos_std)),
            float(rng.normal(0.0, pos_std)),
            float(rng.normal(0.0, rot_std_deg)),
        ))
    return out


def perturb_pairwise(pairwise_t_matrix, dx, dy, dyaw_deg):
    """Left-multiply the partner->ego transform by a small planar offset.

    Returns a new tensor; the input is untouched. pairwise[b, i, j] is the
    i -> j transform, so [b, 1, 0] carries the partner into the ego frame and
    [b, 0, 1] is kept consistent as its inverse.
    """
    if pairwise_t_matrix is None:
        return None
    out = pairwise_t_matrix.clone()
    if dx == 0.0 and dy == 0.0 and dyaw_deg == 0.0:
        return out
    # pairwise_t_matrix is (B, L, L, 4, 4): a 4x4 for every ordered agent pair.
    if out.dim() != 5 or out.shape[1] < 2 or out.shape[2] < 2:
        raise ValueError(
            "perturb_pairwise expected a (B, L, L, 4, 4) pairwise transform, "
            f"got shape {tuple(out.shape)}"
        )

    theta = math.radians(dyaw_deg)
    c, s = math.cos(theta), math.sin(theta)
    offset = torch.eye(4, dtype=out.dtype, device=out.device)
    offset[0, 0], offset[0, 1] = c, -s
    offset[1, 0], offset[1, 1] = s, c
    offset[0, 3], offset[1, 3] = dx, dy

    for b in range(out.shape[0]):
        moved = offset @ out[b, 1, 0].to(offset.dtype)
        out[b, 1, 0] = moved.to(out.dtype)
        try:
            out[b, 0, 1] = torch.linalg.inv(moved).to(out.dtype)
        except RuntimeError:
            pass
    return out


def true_alignment_error(noisy_pairwise, clean_pairwise):
    """Metres of translation error the fusion is actually operating under.

    Mirrors the oracle computation in the model so the probe can be scored
    against it. Returns nan when the clean transform is not in the batch.
    """
    if noisy_pairwise is None or clean_pairwise is None:
        return float("nan")
    try:
        noisy = noisy_pairwise[:, 1, 0].to(torch.float32)
        clean = clean_pairwise[:, 1, 0].to(torch.float32)
        residual = torch.linalg.solve(clean, noisy)
        return float(torch.linalg.norm(residual[:, :2, 3], dim=-1)[0])
    except Exception:
        return float("nan")


def _centres_scores(pred_box_tensor, pred_score):
    """Box centres in the BEV plane and their scores, as numpy."""
    if pred_box_tensor is None or len(pred_box_tensor) == 0:
        return np.zeros((0, 2)), np.zeros((0,))
    boxes = pred_box_tensor.detach().cpu().numpy()
    centres = boxes.reshape(len(boxes), -1, 3)[:, :, :2].mean(axis=1)
    if pred_score is None:
        scores = np.ones(len(centres))
    else:
        scores = pred_score.detach().cpu().numpy().reshape(-1)[:len(centres)]
    return centres, scores


def frame_quality(pred_box_tensor, pred_score, gt_box_tensor, thresh=0.5):
    """Recall and precision for one frame, matched by BEV centre distance.

    Greedy nearest-centre matching at `thresh` metres, the same protocol the
    evaluator uses for mATE. Returns nan recall when the frame has no ground
    truth, so the caller can drop it.
    """
    pred, scores = _centres_scores(pred_box_tensor, pred_score)
    gt, _ = _centres_scores(gt_box_tensor, None)
    if len(gt) == 0:
        return {"recall": float("nan"), "precision": float("nan"),
                "n_gt": 0, "n_pred": len(pred)}
    if len(pred) == 0:
        return {"recall": 0.0, "precision": float("nan"),
                "n_gt": len(gt), "n_pred": 0}

    order = np.argsort(-scores)
    taken = np.zeros(len(gt), dtype=bool)
    hits = 0
    for i in order:
        d = np.linalg.norm(gt - pred[i][None, :], axis=1)
        d[taken] = np.inf
        j = int(np.argmin(d))
        if d[j] <= thresh:
            taken[j] = True
            hits += 1
    return {"recall": hits / len(gt), "precision": hits / len(pred),
            "n_gt": len(gt), "n_pred": len(pred)}


class SensitivityProbe:
    """Turns a set of runs over perturbed partner poses into one number."""

    def __init__(self, lidar_range, cell=0.4):
        self.xmin, self.ymin = float(lidar_range[0]), float(lidar_range[1])
        self.xmax, self.ymax = float(lidar_range[3]), float(lidar_range[4])
        self.cell = float(cell)
        self.nx = max(1, int(math.ceil((self.xmax - self.xmin) / self.cell)))
        self.ny = max(1, int(math.ceil((self.ymax - self.ymin) / self.cell)))

    def _score_map(self, centres, scores):
        m = np.zeros((self.ny, self.nx), dtype=np.float32)
        if len(centres) == 0:
            return m
        ix = np.floor((centres[:, 0] - self.xmin) / self.cell).astype(int)
        iy = np.floor((centres[:, 1] - self.ymin) / self.cell).astype(int)
        keep = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
        for x, y, s in zip(ix[keep], iy[keep], scores[keep]):
            m[y, x] = max(m[y, x], float(s))
        return m

    def score(self, runs):
        """runs: list of (pred_box_tensor, pred_score), the first unperturbed.

        Returns a dict. `map_std` is the headline number: the spread of the
        score map across the perturbations, averaged over the cells any run
        put something in. It is 0 when every perturbation gives the same
        answer and grows as the output becomes pose-dependent.
        """
        parsed = [_centres_scores(b, s) for b, s in runs]
        maps = np.stack([self._score_map(c, s) for c, s in parsed])
        counts = np.array([len(c) for c, _ in parsed], dtype=float)

        touched = maps.max(axis=0) > 0
        n_touched = int(touched.sum())
        map_std = float(maps.std(axis=0)[touched].mean()) if n_touched else 0.0

        ref = parsed[0][0]
        jitter = float("nan")
        if len(ref) and len(parsed) > 1:
            dists = []
            for c, _ in parsed[1:]:
                if len(c) == 0:
                    continue
                d = np.linalg.norm(c[:, None, :] - ref[None, :, :], axis=-1)
                dists.append(float(d.min(axis=1).mean()))
            if dists:
                jitter = float(np.mean(dists))

        return {
            "map_std": map_std,
            "cells_touched": n_touched,
            "n_mean": float(counts.mean()),
            "n_std": float(counts.std()),
            "n_cv": float(counts.std() / counts.mean()) if counts.mean() else 0.0,
            "jitter_m": jitter,
        }
