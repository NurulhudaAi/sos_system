"""
=====================================================================
 train.py  –  Fine-tune YOLOv8-Pose on fall / non-fall dataset
=====================================================================
Dataset layout expected (YOLO pose format):
  data/
    images/
      train/  *.jpg
      val/    *.jpg
    labels/
      train/  *.txt   (YOLO pose format – see below)
      val/    *.txt

Label format per line:
  <class> <cx> <cy> <w> <h>  [x1 y1 c1  x2 y2 c2 ... x17 y17 c17]
  class 0 = person_standing_or_walking
  class 1 = person_FALLING (dangerous)

Recommended public datasets to combine:
  • Le2i Fall Detection  (https://imvia.u-bourgogne.fr/en/database)
  • URFD                 (http://fenix.ur.edu.pl/mkepski/ds/uf.html)
  • FallAllD             (https://falldataset.com)
  • COCO (persons)  → negative examples
"""

from pathlib import Path
import yaml, shutil, random, cv2, numpy as np
from ultralytics import YOLO


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Dataset YAML
# ─────────────────────────────────────────────────────────────────────────────

DATASET_YAML = {
    "path": str(Path("data/processed").resolve()),
    "train": "images/train",
    "val":   "images/val",
    "nc": 2,
    "names": {0: "person", 1: "fall"},
    # 17-kp COCO skeleton
    "kpt_shape": [17, 3],
    "flip_idx": [
        0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15
    ],
}

def write_dataset_yaml(out_path: str = "configs/fall_dataset.yaml"):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(DATASET_YAML, f, default_flow_style=False, allow_unicode=True)
    print(f"✅  Dataset YAML written → {out_path}")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Training  (best settings to maximise recall + minimise false positives)
# ─────────────────────────────────────────────────────────────────────────────

def train(
    data_yaml: str = "configs/fall_dataset.yaml",
    base_model: str = "yolov8l-pose.pt",   # l = better accuracy than n/m
    epochs: int = 150,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",                      # GPU id, or "cpu"
    project: str = "runs/fall_detect",
    name: str    = "exp1",
    resume: bool = False,
):
    """
    Key hyper-parameter decisions
    ─────────────────────────────
    • mosaic=0.5       : blend scenes → harder negatives
    • mixup=0.1        : reduce overfit on small dataset
    • copy_paste=0.1   : synthesise extra fall poses
    • cls=2.0          : higher class loss → penalise FP on "fall" class
    • pose=12.0        : emphasise keypoint accuracy
    • close_mosaic=20  : disable mosaic last 20 epochs for fine-tune stability
    • label_smoothing  : 0.05 prevents overconfident predictions
    • patience=30      : early stopping
    """
    model = YOLO(base_model)

    results = model.train(
        data    = data_yaml,
        epochs  = epochs,
        imgsz   = imgsz,
        batch   = batch,
        device  = device,
        project = project,
        name    = name,
        resume  = resume,

        # ── optimiser ──
        optimizer = "AdamW",
        lr0       = 5e-4,
        lrf       = 0.01,
        warmup_epochs  = 5,
        weight_decay   = 5e-4,
        momentum       = 0.937,

        # ── loss weights ──
        box   = 7.5,
        cls   = 2.0,   # ↑ penalise class errors (crucial for FP reduction)
        dfl   = 1.5,
        pose  = 12.0,  # ↑ strong pose supervision

        # ── augmentation (anti-overfit for small fall datasets) ──
        mosaic      = 0.5,
        mixup       = 0.1,
        copy_paste  = 0.1,
        degrees     = 10,
        translate   = 0.15,
        scale       = 0.6,
        shear       = 2.0,
        perspective = 0.0005,
        flipud      = 0.0,   # no vertical flip (gravity matters!)
        fliplr      = 0.5,
        hsv_h       = 0.015,
        hsv_s       = 0.7,
        hsv_v       = 0.4,
        close_mosaic = 20,

        # ── training tricks ──
        label_smoothing = 0.05,
        patience        = 30,
        plots           = True,
        save_period     = 10,
        amp             = True,   # mixed precision
        multi_scale     = True,
    )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Pseudo-label generator  (bootstrap more training data from raw videos)
# ─────────────────────────────────────────────────────────────────────────────

def extract_frames_from_video(
    video_path: str,
    out_dir: str,
    sample_every: int = 5,
    max_frames: int = 2000,
):
    """Extract frames for manual annotation."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    i, saved = 0, 0
    video_stem = Path(video_path).stem
    while saved < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if i % sample_every == 0:
            fname = out / f"{video_stem}_{i:06d}.jpg"
            cv2.imwrite(str(fname), frame)
            saved += 1
        i += 1
    cap.release()
    print(f"Extracted {saved} frames → {out_dir}")


def pseudo_label_frames(
    frame_dir: str,
    label_dir: str,
    base_model: str = "yolov8l-pose.pt",
    conf: float = 0.6,
):
    """
    Run pre-trained YOLO-pose to create pseudo-labels (standing persons only).
    You then manually review & relabel the fall frames.
    """
    model = YOLO(base_model)
    Path(label_dir).mkdir(parents=True, exist_ok=True)

    images = list(Path(frame_dir).glob("*.jpg"))
    for img_path in images:
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        results = model(img, conf=conf, verbose=False)
        label_lines = []
        for r in results:
            if r.boxes is None:
                continue
            for bi, box in enumerate(r.boxes.xyxy):
                x1, y1, x2, y2 = box.cpu().numpy()
                cx = (x1 + x2) / 2 / w
                cy = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                kps = r.keypoints.data[bi].cpu().numpy()  # (17,3)
                kp_str = " ".join(
                    f"{kp[0]/w:.6f} {kp[1]/h:.6f} {kp[2]:.4f}"
                    for kp in kps
                )
                # class 0 = person (manually change to 1 for falls)
                label_lines.append(
                    f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {kp_str}"
                )
        label_path = Path(label_dir) / (img_path.stem + ".txt")
        label_path.write_text("\n".join(label_lines))

    print(f"Pseudo-labels written for {len(images)} images → {label_dir}")


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Train/val split
# ─────────────────────────────────────────────────────────────────────────────

def split_dataset(
    src_images: str,
    src_labels: str,
    dst_root: str = "data/processed",
    val_ratio: float = 0.15,
    seed: int = 42,
):
    random.seed(seed)
    imgs = sorted(Path(src_images).glob("*.jpg"))
    random.shuffle(imgs)
    n_val = max(1, int(len(imgs) * val_ratio))
    splits = {"val": imgs[:n_val], "train": imgs[n_val:]}

    for split, split_imgs in splits.items():
        img_dst = Path(dst_root) / "images" / split
        lbl_dst = Path(dst_root) / "labels" / split
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)
        for img in split_imgs:
            shutil.copy(img, img_dst / img.name)
            lbl = Path(src_labels) / (img.stem + ".txt")
            if lbl.exists():
                shutil.copy(lbl, lbl_dst / lbl.name)
    print(f"Split complete → {dst_root}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--action", choices=["train", "split", "pseudo"],
                   default="train")
    p.add_argument("--base-model", default="yolov8l-pose.pt")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch",  type=int, default=16)
    p.add_argument("--device", default="0")
    p.add_argument("--data",   default="configs/fall_dataset.yaml")
    args = p.parse_args()

    if args.action == "train":
        yaml_path = write_dataset_yaml(args.data)
        train(
            data_yaml  = yaml_path,
            base_model = args.base_model,
            epochs     = args.epochs,
            batch      = args.batch,
            device     = args.device,
        )
    elif args.action == "split":
        split_dataset("data/raw/images", "data/raw/labels")
    elif args.action == "pseudo":
        pseudo_label_frames("data/raw/images", "data/raw/labels")
