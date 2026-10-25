"""Exact, tie-aware binary ranking metrics without a scikit-learn dependency."""
from __future__ import annotations

import numpy as np


def _pairs(target, score):
    target = np.asarray(target).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    if target.size != score.size or not target.size:
        raise ValueError("Target and score must have equal, nonzero lengths")
    if not np.isfinite(score).all() or not np.isfinite(target).all():
        raise ValueError("Metrics require finite targets and scores")
    return (target > 0).astype(np.int64), score


def roc_auc(target, score):
    target, score = _pairs(target, score)
    positives = int(target.sum())
    negatives = int(target.size - positives)
    if not positives or not negatives:
        return float("nan")
    order = np.argsort(score, kind="stable")
    ranked = score[order]
    starts = np.r_[0, np.flatnonzero(ranked[1:] != ranked[:-1]) + 1]
    counts = np.diff(np.r_[starts, target.size])
    group_positives = np.add.reduceat(target[order], starts)
    average_ranks = starts + (counts + 1.0) / 2.0
    rank_sum = float(np.dot(group_positives, average_ranks))
    return (rank_sum - positives * (positives + 1.0) / 2.0) / (positives * negatives)


def precision_recall_summary(target, score):
    target, score = _pairs(target, score)
    positives = int(target.sum())
    if not positives:
        return 0.0, 0.0, float(np.nextafter(score.max(), np.inf))
    order = np.argsort(-score, kind="stable")
    ranked = score[order]
    ends = np.r_[np.flatnonzero(ranked[1:] != ranked[:-1]), target.size - 1]
    true_positives = np.cumsum(target[order], dtype=np.int64)[ends]
    predicted = ends + 1
    precision = true_positives / predicted
    increments = np.diff(np.r_[0, true_positives])
    ap = float(np.dot(precision, increments) / positives)
    f1 = 2.0 * true_positives / (predicted + positives)
    best = int(np.argmax(f1))
    return ap, float(f1[best]), float(ranked[ends[best]])


def binary_report(target, score, threshold=0.5):
    target, score = _pairs(target, score)
    truth = target.astype(bool)
    prediction = score >= float(threshold)
    tp = int((prediction & truth).sum())
    fp = int((prediction & ~truth).sum())
    fn = int((~prediction & truth).sum())
    ap, best_f1, best_threshold = precision_recall_summary(target, score)
    return {
        "pixels": int(target.size), "positive_pixels": int(target.sum()),
        "prevalence": float(target.mean()), "roc_auc": roc_auc(target, score),
        "ap": ap, "best_f1_diagnostic": best_f1,
        "best_threshold_diagnostic": best_threshold, "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "fp_per_mpix": fp / target.size * 1e6,
        "brier": float(np.mean((score - target) ** 2)),
    }
