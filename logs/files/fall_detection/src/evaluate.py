"""
=====================================================================
 evaluate.py  –  Threshold tuning to minimize false positives
=====================================================================
Run after training to find the optimal danger_score_threshold
on your validation videos.  Outputs ROC curve + confusion matrix.
"""

import json, cv2, numpy as np
from pathlib import Path
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, classification_report, f1_score
)
import matplotlib.pyplot as plt
from fall_detector import DangerousFallDetector, FallThresholds


def evaluate_on_video(
    video_path: str,
    gt_fall_intervals: list[tuple[float, float]],   # [(start_sec, end_sec), ...]
    model_path: str = "runs/fall_detect/exp1/weights/best.pt",
    score_threshold: float = 0.60,
    device: str = "cpu",
) -> dict:
    """
    Parameters
    ----------
    gt_fall_intervals : ground-truth fall intervals in seconds
        e.g. [(12.5, 15.0), (40.0, 43.5)]
    """
    th = FallThresholds(danger_score_threshold=score_threshold)
    detector = DangerousFallDetector(model_path=model_path, thresholds=th, device=device)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    detector.fps_estimate = fps

    y_true, y_score = [], []
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        _, events = detector.process_frame(frame)
        t_sec = frame_idx / fps

        gt_label = int(any(s <= t_sec <= e for s, e in gt_fall_intervals))
        pred_score = max((e["danger_score"] for e in events), default=0.0)

        y_true.append(gt_label)
        y_score.append(pred_score)
        frame_idx += 1

    cap.release()

    y_true  = np.array(y_true)
    y_score = np.array(y_score)
    y_pred  = (y_score >= score_threshold).astype(int)

    cm  = confusion_matrix(y_true, y_pred)
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)

    return {
        "roc_auc":   roc_auc,
        "fpr":       fpr.tolist(),
        "tpr":       tpr.tolist(),
        "thresholds": thresholds.tolist(),
        "confusion_matrix": cm.tolist(),
        "y_true":   y_true.tolist(),
        "y_score":  y_score.tolist(),
        "report":   classification_report(y_true, y_pred,
                                          target_names=["normal", "fall"]),
    }


def find_optimal_threshold(
    y_true: list, y_score: list,
    beta: float = 2.0,   # F-beta  (β>1 weights recall higher than precision)
) -> float:
    """
    Choose threshold that maximises F-beta score.
    β=2 : we care more about catching real falls (recall)
         than about perfect precision.
    Use β=0.5 if you want to aggressively reduce false positives.
    """
    y_true  = np.array(y_true)
    y_score = np.array(y_score)

    best_thresh, best_fb = 0.5, -1.0
    for t in np.arange(0.3, 0.95, 0.01):
        y_pred = (y_score >= t).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()

        precision = tp / (tp + fp + 1e-9)
        recall    = tp / (tp + fn + 1e-9)
        fb = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + 1e-9)

        if fb > best_fb:
            best_fb, best_thresh = fb, t

    print(f"Optimal threshold = {best_thresh:.2f}  (F-{beta:.1f} = {best_fb:.3f})")
    return best_thresh


def plot_roc(fpr, tpr, roc_auc, out="results/roc_curve.png"):
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color="#E63946", lw=2,
             label=f"ROC curve  (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlim([0, 1])
    plt.ylim([0, 1.02])
    plt.xlabel("False Positive Rate (Stumble falsely flagged)")
    plt.ylabel("True Positive Rate (Dangerous falls caught)")
    plt.title("Fall Detector  –  ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"ROC saved → {out}")


def plot_threshold_sweep(y_true, y_score, out="results/threshold_sweep.png"):
    """Plot Precision / Recall / F1 vs threshold."""
    thresholds = np.arange(0.1, 0.99, 0.01)
    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        y_pred = (np.array(y_score) >= t).astype(int)
        tp = ((y_pred == 1) & (np.array(y_true) == 1)).sum()
        fp = ((y_pred == 1) & (np.array(y_true) == 0)).sum()
        fn = ((y_pred == 0) & (np.array(y_true) == 1)).sum()
        p  = tp / (tp + fp + 1e-9)
        r  = tp / (tp + fn + 1e-9)
        f1 = 2 * p * r / (p + r + 1e-9)
        precisions.append(p); recalls.append(r); f1s.append(f1)

    plt.figure(figsize=(9, 5))
    plt.plot(thresholds, precisions, label="Precision", color="#2196F3", lw=2)
    plt.plot(thresholds, recalls,    label="Recall",    color="#4CAF50", lw=2)
    plt.plot(thresholds, f1s,        label="F1",        color="#FF5722", lw=2)
    best_t = thresholds[np.argmax(f1s)]
    plt.axvline(best_t, linestyle="--", color="gray", label=f"Best F1 @ {best_t:.2f}")
    plt.xlabel("Danger Score Threshold")
    plt.ylabel("Score")
    plt.title("Precision / Recall / F1  vs  Threshold")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"Threshold sweep saved → {out}")


if __name__ == "__main__":
    # ── Example usage ──────────────────────────────────────────────────────
    # Replace with your actual validation video + annotated fall intervals
    VIDEO = "data/val_video.mp4"
    GT_FALLS = [(12.0, 15.5), (38.0, 41.0)]   # seconds

    result = evaluate_on_video(VIDEO, GT_FALLS)
    print(result["report"])

    opt_thresh = find_optimal_threshold(result["y_true"], result["y_score"])
    plot_roc(result["fpr"], result["tpr"], result["roc_auc"])
    plot_threshold_sweep(result["y_true"], result["y_score"])

    # Save optimal threshold back to config
    import yaml
    cfg_path = "configs/thresholds.yaml"
    cfg = {"danger_score_threshold": round(float(opt_thresh), 3)}
    Path(cfg_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cfg_path, "w") as f:
        yaml.dump(cfg, f)
    print(f"Saved optimal threshold → {cfg_path}")
