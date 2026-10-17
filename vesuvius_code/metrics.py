import torch


def _valid_boolean_mask(reference, valid_mask):
    if valid_mask is None:
        return torch.ones_like(reference, dtype=torch.bool)
    if valid_mask.shape != reference.shape:
        valid_mask = torch.broadcast_to(valid_mask, reference.shape)
    return valid_mask.to(device=reference.device, dtype=torch.bool)


def binary_segmentation_metrics(
    probs, target, threshold=0.5, eps=1e-8, valid_mask=None
):
    """
    Metrics for sparse ink segmentation.

    fp_per_mpix is intentionally included because visual Dice can look good while
    a full scroll is polluted by thousands of false-positive pixels.
    """

    pred = probs >= threshold
    truth = target >= 0.5
    valid = _valid_boolean_mask(pred, valid_mask)

    tp = int((pred & truth & valid).sum().item())
    fp = int((pred & ~truth & valid).sum().item())
    fn = int((~pred & truth & valid).sum().item())
    tn = int((~pred & ~truth & valid).sum().item())
    valid_pixels = int(valid.sum().item())
    total = max(1, valid_pixels)

    precision = tp / max(eps, tp + fp)
    recall = tp / max(eps, tp + fn)
    f1 = (2.0 * precision * recall) / max(eps, precision + recall)
    fp_per_mpix = fp / total * 1_000_000

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "valid_pixels": valid_pixels,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fp_per_mpix": float(fp_per_mpix),
    }


def aggregate_binary_segmentation_metrics(metric_rows, eps=1e-8):
    tp = sum(int(row["tp"]) for row in metric_rows)
    fp = sum(int(row["fp"]) for row in metric_rows)
    fn = sum(int(row["fn"]) for row in metric_rows)
    tn = sum(int(row["tn"]) for row in metric_rows)
    valid_pixels = sum(
        int(row.get("valid_pixels", row["tp"] + row["fp"] + row["fn"] + row["tn"]))
        for row in metric_rows
    )
    total = max(1, valid_pixels)

    precision = tp / max(eps, tp + fp)
    recall = tp / max(eps, tp + fn)
    f1 = (2.0 * precision * recall) / max(eps, precision + recall)
    fp_per_mpix = fp / total * 1_000_000

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "valid_pixels": valid_pixels,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "fp_per_mpix": float(fp_per_mpix),
    }


def probability_calibration_metrics(probs, target, valid_mask=None, num_bins=15):
    if int(num_bins) <= 0:
        raise ValueError("num_bins must be positive")
    probs = probs.to(dtype=torch.float32)
    target = target.to(device=probs.device, dtype=torch.float32)
    valid = _valid_boolean_mask(probs, valid_mask)
    selected_probs = probs[valid]
    selected_target = target[valid]
    count = int(selected_probs.numel())
    if count == 0:
        return {
            "valid_pixels": 0,
            "brier": float("nan"),
            "ece": float("nan"),
            "mean_probability": float("nan"),
            "positive_rate": float("nan"),
        }

    brier = torch.mean((selected_probs - selected_target) ** 2)
    boundaries = torch.linspace(0.0, 1.0, int(num_bins) + 1, device=probs.device)
    ece = torch.zeros((), device=probs.device)
    for index in range(int(num_bins)):
        lower = boundaries[index]
        upper = boundaries[index + 1]
        if index == int(num_bins) - 1:
            in_bin = (selected_probs >= lower) & (selected_probs <= upper)
        else:
            in_bin = (selected_probs >= lower) & (selected_probs < upper)
        if bool(in_bin.any()):
            confidence = selected_probs[in_bin].mean()
            accuracy = selected_target[in_bin].mean()
            ece = ece + in_bin.float().mean() * torch.abs(confidence - accuracy)

    return {
        "valid_pixels": count,
        "brier": float(brier.item()),
        "ece": float(ece.item()),
        "mean_probability": float(selected_probs.mean().item()),
        "positive_rate": float(selected_target.mean().item()),
    }


def average_precision(probs, target, valid_mask=None):
    """Compute non-interpolated average precision over valid pixels."""

    probs = probs.to(dtype=torch.float32)
    target = target.to(device=probs.device, dtype=torch.float32)
    valid = _valid_boolean_mask(probs, valid_mask)
    selected_probs = probs[valid].reshape(-1)
    selected_target = (target[valid].reshape(-1) >= 0.5).to(torch.float32)
    valid_pixels = int(selected_probs.numel())
    positive_pixels = int(selected_target.sum().item())
    if valid_pixels == 0 or positive_pixels == 0:
        return {
            "valid_pixels": valid_pixels,
            "positive_pixels": positive_pixels,
            "average_precision": float("nan"),
        }

    order = torch.argsort(selected_probs, descending=True, stable=True)
    ranked_probs = selected_probs[order]
    ranked_target = selected_target[order]
    cumulative_true = torch.cumsum(ranked_target, dim=0)
    ranks = torch.arange(
        1,
        valid_pixels + 1,
        device=probs.device,
        dtype=torch.float32,
    )
    group_end = torch.ones(valid_pixels, device=probs.device, dtype=torch.bool)
    if valid_pixels > 1:
        group_end[:-1] = ranked_probs[:-1] != ranked_probs[1:]
    true_at_threshold = cumulative_true[group_end]
    precision_at_threshold = true_at_threshold / ranks[group_end]
    positive_increments = torch.diff(
        torch.cat((torch.zeros(1, device=probs.device), true_at_threshold))
    )
    score = (precision_at_threshold * positive_increments).sum() / positive_pixels
    return {
        "valid_pixels": valid_pixels,
        "positive_pixels": positive_pixels,
        "average_precision": float(score.item()),
    }


def threshold_sweep_metrics(
    probs,
    target,
    thresholds=None,
    valid_mask=None,
):
    """Select a validation threshold by F1, then fewer false positives."""

    if thresholds is None:
        thresholds = [index / 20.0 for index in range(1, 20)]
    candidates = [float(value) for value in thresholds]
    if not candidates:
        raise ValueError("thresholds must not be empty")
    if any(value < 0.0 or value > 1.0 for value in candidates):
        raise ValueError("thresholds must be between zero and one")

    rows = []
    for threshold in candidates:
        row = binary_segmentation_metrics(
            probs,
            target,
            threshold=threshold,
            valid_mask=valid_mask,
        )
        rows.append({"threshold": threshold, **row})
    return max(
        rows,
        key=lambda row: (row["f1"], -row["fp"], row["threshold"]),
    )


def format_metrics(metrics):
    return (
        f"precision={metrics['precision']:.4f} "
        f"recall={metrics['recall']:.4f} "
        f"f1={metrics['f1']:.4f} "
        f"fp/Mpix={metrics['fp_per_mpix']:.1f}"
    )
