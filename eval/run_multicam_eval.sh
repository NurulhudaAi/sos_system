#!/bin/bash
# eval/run_multicam_eval.sh — End-to-end MultiCam Fall Dataset evaluation
set -e

DATASET_DIR="/Users/nurulhudaadamishaq/Downloads/dataset"
LABELS_DIR="eval/labels"
VIDEOS_DIR="eval/videos"
OUT_DIR="eval/report_multicam"

echo "============================================================"
echo "MultiCam Fall Dataset Evaluation Pipeline"
echo "============================================================"

echo ""
echo "Step 1: Generating labels from data_tuple3.csv ..."
python3 eval/generate_multicam_labels.py \
    --dataset-dir "$DATASET_DIR" \
    --cam 1 \
    --out-labels-dir "$LABELS_DIR" \
    --out-videos-dir "$VIDEOS_DIR" \
    --link-mode symlink

echo ""
echo "Step 2: Running evaluation ..."
python3 eval/eval_augmented.py \
    --videos-dir "$VIDEOS_DIR" \
    --labels-dir "$LABELS_DIR" \
    --detector-url http://127.0.0.1:8000 \
    --out-dir "$OUT_DIR" \
    --sample-fps 5 \
    --tolerance-sec 5.0 \
    --augmentations all

echo ""
echo "============================================================"
echo "Done! Reports saved to: $OUT_DIR/"
echo "============================================================"
