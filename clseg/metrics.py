"""Dataset-level confusion metrics; undefined class scores are NaN, never perfect."""
import numpy as np
from scipy.ndimage import binary_erosion
import torch
import torch.nn.functional as F


def divide(a, b):
    return np.divide(a, b, out=np.full(np.shape(a), np.nan, dtype=float), where=np.asarray(b) != 0)


def mean_defined(a):
    a = np.asarray(a)
    return float(np.nanmean(a)) if np.isfinite(a).any() else float("nan")


class SegMetrics:
    def __init__(self, classes, ignore=255, boundary_ratio=.02, eval_classes=None, compute_boundary=True):
        self.c, self.ignore, self.ratio = classes, ignore, boundary_ratio
        self.compute_boundary = compute_boundary
        self.indices = list(range(classes)) if eval_classes is None else eval_classes
        self.cm = np.zeros((classes, classes), dtype=np.int64)
        self.bi = np.zeros(classes, dtype=np.int64)
        self.bu = np.zeros(classes, dtype=np.int64)

    def update(self, prediction, target):
        p, y = np.asarray(prediction), np.asarray(target)
        if p.ndim == 2:
            p, y = p[None], y[None]
        for pred, truth in zip(p, y):
            valid = truth != self.ignore
            if not valid.any():
                continue
            self.cm += np.bincount(self.c * truth[valid] + pred[valid], minlength=self.c ** 2).reshape(self.c, self.c)
            if not self.compute_boundary:
                continue
            radius = max(1, round(self.ratio * np.hypot(*truth.shape)))
            # Remove an ignore-neighbourhood so artificial void borders are not scored.
            safe = binary_erosion(valid, structure=np.ones((3, 3)), iterations=radius, border_value=1)
            for c in self.indices:
                gt, pr = (truth == c) & valid, (pred == c) & valid
                gb = gt & ~binary_erosion(gt, structure=np.ones((3, 3)), iterations=radius, border_value=0) & safe
                pb = pr & ~binary_erosion(pr, structure=np.ones((3, 3)), iterations=radius, border_value=0) & safe
                self.bi[c] += (gb & pb).sum()
                self.bu[c] += (gb | pb).sum()

    def compute(self):
        tp = np.diag(self.cm)
        gt, pr = self.cm.sum(1), self.cm.sum(0)
        union = gt + pr - tp
        iou, f1, biou = divide(tp, union), divide(2 * tp, gt + pr), divide(self.bi, self.bu)
        idx = self.indices
        excluded = sorted(set(range(self.c)) - set(idx))
        iou[excluded], f1[excluded], biou[excluded] = np.nan, np.nan, np.nan
        return {"accuracy": float(divide(tp[idx].sum(), gt[idx].sum())), "miou": mean_defined(iou[idx]),
                "iou": float(divide(tp[idx].sum(), union[idx].sum())), "biou": mean_defined(biou[idx]),
                "dice": mean_defined(f1[idx]), "macro_f1": mean_defined(f1[idx]),
                "class_iou": iou.tolist(), "class_f1": f1.tolist(), "class_biou": biou.tolist(),
                "class_support": gt.tolist(), "confusion_matrix": self.cm.tolist(), "evaluated_classes": idx}


def segmentation_loss(logits, target, cfg, ignore):
    valid = target != ignore
    if not valid.any():
        return None
    options = cfg["training"]["loss"]
    if not options["cross_entropy"] and not options["dice"]:
        raise ValueError("At least one supervised loss must be enabled")
    loss = logits.float().sum() * 0
    if options["cross_entropy"]:
        weights = options.get("class_weights")
        weights = torch.as_tensor(weights, dtype=logits.dtype, device=logits.device) if weights else None
        loss = loss + F.cross_entropy(logits, target, weight=weights, ignore_index=ignore)
    if options["dice"]:
        safe = target.masked_fill(~valid, 0)
        onehot = F.one_hot(safe, logits.shape[1]).permute(0, 3, 1, 2).float() * valid[:, None]
        probs = logits.softmax(1) * valid[:, None]
        dims = (0, 2, 3)
        positives = onehot.sum(dims) > 0
        dice = (2 * (probs * onehot).sum(dims) + 1e-6) / (probs.sum(dims) + onehot.sum(dims) + 1e-6)
        loss = loss + options["dice_weight"] * (1 - dice[positives]).mean()
    return loss


def continual_scores(matrix):
    """R[t,j] is performance on fixed task j after stage t; upper triangle NaN."""
    r = np.asarray(matrix, dtype=float)
    t = len(r) - 1
    if t <= 0:
        return {"average_seen_miou": mean_defined(r[-1]), "forgetting": None, "backward_transfer": None}
    forgetting, backward = [], []
    for j in range(t):
        history = r[j:t, j]
        if np.isfinite(history).any() and np.isfinite(r[t, j]):
            forgetting.append(np.nanmax(history) - r[t, j])
            backward.append(r[t, j] - r[j, j])
    return {"average_seen_miou": mean_defined(r[t, :t + 1]), "forgetting": mean_defined(forgetting), "backward_transfer": mean_defined(backward)}
