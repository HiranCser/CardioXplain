import json
from dataclasses import dataclass

import numpy as np


CLASS_NAMES = [
    "normal_contraction",
    "reduced_contraction",
    "severe_dysfunction",
]


LABEL_TO_TEXT = {
    0: CLASS_NAMES[0],
    1: CLASS_NAMES[1],
    2: CLASS_NAMES[2],
}


TEXT_TO_LABEL = {v: k for k, v in LABEL_TO_TEXT.items()}


def ef_to_severity_label(ef_pct, normal_threshold=50.0, severe_threshold=30.0):
    """
    Convert EF (%) to 3-class contraction label.
    0: normal_contraction (EF >= normal_threshold)
    1: reduced_contraction (severe_threshold <= EF < normal_threshold)
    2: severe_dysfunction (EF < severe_threshold)
    """
    ef = float(ef_pct)
    if ef >= float(normal_threshold):
        return 0
    if ef < float(severe_threshold):
        return 2
    return 1


def softmax_np(logits, temperature=1.0):
    t = max(float(temperature), 1e-6)
    z = np.asarray(logits, dtype=np.float64) / t
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.clip(np.sum(exp_z, axis=1, keepdims=True), 1e-12, None)


def nll_np(probs, labels):
    y = np.asarray(labels, dtype=np.int64)
    p = np.clip(np.asarray(probs, dtype=np.float64), 1e-12, 1.0)
    return float(-np.log(p[np.arange(y.shape[0]), y]).mean())


def accuracy_np(pred, labels):
    p = np.asarray(pred, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int64)
    if y.size == 0:
        return float("nan")
    return float((p == y).mean())


def macro_f1_np(pred, labels, n_classes=3):
    p = np.asarray(pred, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int64)
    if y.size == 0:
        return float("nan")

    f1s = []
    for c in range(int(n_classes)):
        tp = float(np.sum((p == c) & (y == c)))
        fp = float(np.sum((p == c) & (y != c)))
        fn = float(np.sum((p != c) & (y == c)))

        precision = tp / (tp + fp + 1e-12)
        recall = tp / (tp + fn + 1e-12)
        f1 = 2.0 * precision * recall / (precision + recall + 1e-12)
        f1s.append(f1)

    return float(np.mean(f1s))


def confusion_matrix_np(pred, labels, n_classes=3):
    p = np.asarray(pred, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int64)
    cm = np.zeros((int(n_classes), int(n_classes)), dtype=np.int64)
    for yy, pp in zip(y.tolist(), p.tolist()):
        if 0 <= yy < n_classes and 0 <= pp < n_classes:
            cm[yy, pp] += 1
    return cm


@dataclass
class Stage6SimilarityEngine:
    class_names: tuple = tuple(CLASS_NAMES)

    def __post_init__(self):
        self.class_names = tuple(self.class_names)
        self.n_classes = len(self.class_names)
        self.prototypes = None
        self.class_priors = None
        self.feature_mean = None
        self.feature_std = None

    def fit(self, features, labels):
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.int64)
        if x.ndim != 2:
            raise ValueError("features must be a 2D array")
        if y.ndim != 1 or y.shape[0] != x.shape[0]:
            raise ValueError("labels must be shape (N,) and match features")
        if x.shape[0] == 0:
            raise ValueError("cannot fit Stage6 similarity engine with zero samples")

        self.feature_mean = np.nanmean(x, axis=0)
        self.feature_std = np.nanstd(x, axis=0)
        self.feature_std = np.where(self.feature_std < 1e-8, 1.0, self.feature_std)

        x = np.where(np.isfinite(x), x, self.feature_mean)
        xz = (x - self.feature_mean) / self.feature_std

        self.prototypes = np.zeros((self.n_classes, xz.shape[1]), dtype=np.float64)
        self.class_priors = np.zeros((self.n_classes,), dtype=np.float64)

        global_proto = np.mean(xz, axis=0)

        for c in range(self.n_classes):
            mask = y == c
            self.class_priors[c] = float(np.mean(mask))
            if np.any(mask):
                self.prototypes[c] = np.mean(xz[mask], axis=0)
            else:
                self.prototypes[c] = global_proto

        prior_sum = np.sum(self.class_priors)
        if prior_sum <= 0:
            self.class_priors[:] = 1.0 / float(self.n_classes)
        else:
            self.class_priors /= prior_sum

    def _check_fitted(self):
        if self.prototypes is None or self.feature_mean is None or self.feature_std is None:
            raise RuntimeError("Stage6SimilarityEngine is not fitted")

    def transform(self, features):
        self._check_fitted()
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2:
            raise ValueError("features must be 2D")
        x = np.where(np.isfinite(x), x, self.feature_mean)
        return (x - self.feature_mean) / self.feature_std

    def predict_logits(self, features):
        self._check_fitted()
        xz = self.transform(features)
        diff = xz[:, None, :] - self.prototypes[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)

        prior = np.clip(self.class_priors, 1e-12, 1.0)
        logits = -dist2 + np.log(prior[None, :])
        return logits

    def predict_proba(self, features, temperature=1.0):
        logits = self.predict_logits(features)
        return softmax_np(logits, temperature=temperature)

    def predict(self, features, temperature=1.0):
        probs = self.predict_proba(features, temperature=temperature)
        return np.argmax(probs, axis=1)

    def save_npz(self, path):
        self._check_fitted()
        np.savez_compressed(
            path,
            class_names=np.array(self.class_names, dtype=object),
            prototypes=self.prototypes,
            class_priors=self.class_priors,
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
        )

    @classmethod
    def load_npz(cls, path):
        obj = np.load(path, allow_pickle=True)
        inst = cls(class_names=tuple(obj["class_names"].tolist()))
        inst.prototypes = obj["prototypes"]
        inst.class_priors = obj["class_priors"]
        inst.feature_mean = obj["feature_mean"]
        inst.feature_std = obj["feature_std"]
        return inst


@dataclass
class Stage7UncertaintyCalibrator:
    temperature: float = 1.0
    fusion_alpha: float = 0.5
    q90_abs_error: float = 10.0
    q95_abs_error: float = 15.0

    def fit_temperature(self, val_logits, val_labels, grid=None):
        logits = np.asarray(val_logits, dtype=np.float64)
        labels = np.asarray(val_labels, dtype=np.int64)

        if logits.shape[0] == 0:
            self.temperature = 1.0
            return

        if grid is None:
            grid = np.linspace(0.5, 5.0, 91)

        best_t = 1.0
        best_nll = float("inf")

        for t in grid:
            probs = softmax_np(logits, temperature=float(t))
            loss = nll_np(probs, labels)
            if loss < best_nll:
                best_nll = loss
                best_t = float(t)

        self.temperature = best_t

    def fit_fusion_alpha(self, ef_stage123_pct, ef_stage5_pct, ef_gt_pct, grid=None):
        ef1 = np.asarray(ef_stage123_pct, dtype=np.float64)
        ef5 = np.asarray(ef_stage5_pct, dtype=np.float64)
        gt = np.asarray(ef_gt_pct, dtype=np.float64)

        if ef1.shape[0] == 0:
            self.fusion_alpha = 0.5
            return

        ef5 = np.where(np.isfinite(ef5), ef5, ef1)

        if grid is None:
            grid = np.linspace(0.0, 1.0, 51)

        best_a = 0.5
        best_mae = float("inf")

        for a in grid:
            fused = float(a) * ef1 + (1.0 - float(a)) * ef5
            mae = float(np.mean(np.abs(fused - gt)))
            if mae < best_mae:
                best_mae = mae
                best_a = float(a)

        self.fusion_alpha = best_a

    def fuse_ef(self, ef_stage123_pct, ef_stage5_pct):
        ef1 = np.asarray(ef_stage123_pct, dtype=np.float64)
        ef5 = np.asarray(ef_stage5_pct, dtype=np.float64)
        ef5 = np.where(np.isfinite(ef5), ef5, ef1)
        return self.fusion_alpha * ef1 + (1.0 - self.fusion_alpha) * ef5

    def fit_intervals(self, fused_ef_pct, ef_gt_pct):
        fused = np.asarray(fused_ef_pct, dtype=np.float64)
        gt = np.asarray(ef_gt_pct, dtype=np.float64)

        if fused.shape[0] == 0:
            self.q90_abs_error = 10.0
            self.q95_abs_error = 15.0
            return

        abs_err = np.abs(fused - gt)
        self.q90_abs_error = float(np.quantile(abs_err, 0.90))
        self.q95_abs_error = float(np.quantile(abs_err, 0.95))

    def fit(self, val_logits, val_labels, ef_stage123_pct, ef_stage5_pct, ef_gt_pct):
        self.fit_temperature(val_logits, val_labels)
        self.fit_fusion_alpha(ef_stage123_pct, ef_stage5_pct, ef_gt_pct)
        fused = self.fuse_ef(ef_stage123_pct, ef_stage5_pct)
        self.fit_intervals(fused, ef_gt_pct)

    def calibrated_proba(self, logits):
        return softmax_np(np.asarray(logits, dtype=np.float64), temperature=self.temperature)

    def intervals(self, fused_ef_pct):
        ef = np.asarray(fused_ef_pct, dtype=np.float64)
        lo90 = ef - self.q90_abs_error
        hi90 = ef + self.q90_abs_error
        lo95 = ef - self.q95_abs_error
        hi95 = ef + self.q95_abs_error
        return lo90, hi90, lo95, hi95

    def save_json(self, path):
        payload = {
            "temperature": float(self.temperature),
            "fusion_alpha": float(self.fusion_alpha),
            "q90_abs_error": float(self.q90_abs_error),
            "q95_abs_error": float(self.q95_abs_error),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    @classmethod
    def load_json(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return cls(
            temperature=float(d.get("temperature", 1.0)),
            fusion_alpha=float(d.get("fusion_alpha", 0.5)),
            q90_abs_error=float(d.get("q90_abs_error", 10.0)),
            q95_abs_error=float(d.get("q95_abs_error", 15.0)),
        )
