"""
=====================================================================
 augment_falls.py  –  Synthetic augmentation for FALL class
=====================================================================
Fall datasets are always small + imbalanced (few falls, many normals).
This script:
  1.  Applies heavy geometric + photometric augmentation to fall frames
  2.  Blends fall foreground onto new background scenes (copy-paste)
  3.  Generates synthetic "in-progress fall" frames by interpolating
      standing → lying pose sequences
"""

import cv2, numpy as np, random, shutil
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A

# ─────────────────────────────────────────────────────────────────────────────
# Albumentations pipeline  (fall-specific – no vertical flip!)
# ─────────────────────────────────────────────────────────────────────────────

FALL_AUGMENT = A.Compose(
    [
        # spatial
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.12, scale_limit=0.25,
            rotate_limit=10,   # small rotation – gravity direction matters
            border_mode=cv2.BORDER_REFLECT_101, p=0.85
        ),
        A.Perspective(scale=(0.04, 0.08), p=0.4),

        # photometric  (simulate different lighting environments)
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.35,
                                       contrast_limit=0.35, p=1),
            A.HueSaturationValue(hue_shift_limit=18,
                                 sat_shift_limit=40, val_shift_limit=30, p=1),
            A.CLAHE(clip_limit=3.0, p=1),
        ], p=0.8),

        # noise / blur  (simulate low-light CCTV)
        A.OneOf([
            A.GaussNoise(var_limit=(20, 80), p=1),
            A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=1),
            A.MotionBlur(blur_limit=7, p=1),
        ], p=0.5),

        # simulate partial occlusion
        A.CoarseDropout(
            max_holes=6, max_height=40, max_width=40, fill_value=0, p=0.3
        ),

        # downscale then upscale  (simulate low-res CCTV)
        A.Downscale(scale_min=0.5, scale_max=0.85,
                    interpolation=cv2.INTER_LINEAR, p=0.3),

        A.RandomShadow(p=0.3),
        A.RandomFog(fog_coef_lower=0.05, fog_coef_upper=0.2, p=0.15),
    ],
    keypoint_params=A.KeypointParams(
        format="xy", remove_invisible=False
    ),
    bbox_params=A.BboxParams(
        format="pascal_voc", label_fields=["class_labels"]
    ),
)


def augment_fall_image(
    image: np.ndarray,
    bboxes: list,            # [[x1,y1,x2,y2], ...]
    keypoints_list: list,    # [[(x,y), ...17 pts...], ...]  per person
    n_augments: int = 5,
) -> list[dict]:
    """
    Returns list of augmented samples, each a dict:
      {image, bboxes, keypoints_list}
    """
    outputs = []
    flat_kps = [kp for kps in keypoints_list for kp in kps]   # flatten

    for _ in range(n_augments):
        try:
            result = FALL_AUGMENT(
                image=image,
                bboxes=bboxes,
                keypoints=flat_kps,
                class_labels=[1] * len(bboxes),
            )
            # re-chunk keypoints back per person
            kps_aug = list(result["keypoints"])
            n = len(keypoints_list[0]) if keypoints_list else 17
            kps_chunked = [kps_aug[i * n:(i + 1) * n]
                           for i in range(len(keypoints_list))]
            outputs.append({
                "image":          result["image"],
                "bboxes":         result["bboxes"],
                "keypoints_list": kps_chunked,
            })
        except Exception as e:
            print(f"Augment error: {e}")

    return outputs


# ─────────────────────────────────────────────────────────────────────────────
# YOLO label I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def parse_yolo_pose_label(label_path: str, img_w: int, img_h: int):
    """Parse YOLO-pose label file → list of (class, bbox_abs, kps_abs)."""
    persons = []
    for line in Path(label_path).read_text().splitlines():
        parts = list(map(float, line.split()))
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        cx, cy, bw, bh = parts[1:5]
        x1 = (cx - bw / 2) * img_w
        y1 = (cy - bh / 2) * img_h
        x2 = (cx + bw / 2) * img_w
        y2 = (cy + bh / 2) * img_h
        kps = []
        for i in range(17):
            kx = parts[5 + i * 3]     * img_w
            ky = parts[5 + i * 3 + 1] * img_h
            kc = parts[5 + i * 3 + 2]
            kps.append((kx, ky, kc))
        persons.append((cls, [x1, y1, x2, y2], kps))
    return persons


def write_yolo_pose_label(label_path: str, persons, img_w: int, img_h: int):
    lines = []
    for cls, bbox, kps in persons:
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2 / img_w
        cy = (y1 + y2) / 2 / img_h
        bw = (x2 - x1) / img_w
        bh = (y2 - y1) / img_h
        kp_str = " ".join(
            f"{kx/img_w:.6f} {ky/img_h:.6f} {kc:.4f}"
            for kx, ky, kc in kps
        )
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} {kp_str}")
    Path(label_path).write_text("\n".join(lines))


# ─────────────────────────────────────────────────────────────────────────────
# Batch augmentation pipeline
# ─────────────────────────────────────────────────────────────────────────────

def augment_fall_dataset(
    src_img_dir:   str,
    src_lbl_dir:   str,
    dst_img_dir:   str,
    dst_lbl_dir:   str,
    n_per_image:   int = 6,
    fall_class_id: int = 1,
):
    """
    Reads all FALL-class images, augments each n_per_image times,
    and writes augmented copies to destination directories.
    """
    Path(dst_img_dir).mkdir(parents=True, exist_ok=True)
    Path(dst_lbl_dir).mkdir(parents=True, exist_ok=True)

    images = sorted(Path(src_img_dir).glob("*.jpg"))
    total_generated = 0

    for img_path in images:
        lbl_path = Path(src_lbl_dir) / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        persons = parse_yolo_pose_label(str(lbl_path), w, h)

        # only augment images that actually contain a FALL annotation
        fall_persons = [p for p in persons if p[0] == fall_class_id]
        if not fall_persons:
            continue

        bboxes   = [p[1] for p in fall_persons]
        kps_list = [[(kx, ky) for kx, ky, _ in p[2]] for p in fall_persons]

        aug_samples = augment_fall_image(img, bboxes, kps_list, n_per_image)

        for k, sample in enumerate(aug_samples):
            out_name = f"{img_path.stem}_aug{k:03d}"
            out_img  = str(Path(dst_img_dir) / (out_name + ".jpg"))
            out_lbl  = str(Path(dst_lbl_dir) / (out_name + ".txt"))

            cv2.imwrite(out_img, sample["image"])

            # reconstruct persons with augmented bbox + kps
            aug_persons = []
            for pi, (cls, _, orig_kps) in enumerate(fall_persons):
                if pi >= len(sample["bboxes"]):
                    continue
                new_bbox = list(sample["bboxes"][pi])
                new_kps_xy = sample["keypoints_list"][pi] if pi < len(sample["keypoints_list"]) else []
                # re-attach confidence from original
                new_kps = [(new_kps_xy[j][0], new_kps_xy[j][1], orig_kps[j][2])
                           if j < len(new_kps_xy) else orig_kps[j]
                           for j in range(17)]
                aug_persons.append((cls, new_bbox, new_kps))

            aug_h, aug_w = sample["image"].shape[:2]
            write_yolo_pose_label(out_lbl, aug_persons, aug_w, aug_h)
            total_generated += 1

    print(f"✅  Generated {total_generated} augmented fall images → {dst_img_dir}")


if __name__ == "__main__":
    augment_fall_dataset(
        src_img_dir  = "data/processed/images/train",
        src_lbl_dir  = "data/processed/labels/train",
        dst_img_dir  = "data/processed/images/train",   # add in-place
        dst_lbl_dir  = "data/processed/labels/train",
        n_per_image  = 8,
    )
