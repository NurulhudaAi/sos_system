import cv2, time
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def preprocess(frame, w, h):
    if frame.shape[1]!=w or frame.shape[0]!=h:
        frame = cv2.resize(frame,(w,h))
    return frame

# ── Font paths ────────────────────────────────────────────────────────────────
# BUG FIX: NotoSansThai ไม่มี Latin glyph → ใช้สอง font แยกกัน
# - NotoSans     → Latin / ASCII / ตัวเลข / สัญลักษณ์
# - NotoSansThai → ภาษาไทย (U+0E00–U+0E7F)

_LATIN_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Tahoma.ttf",
    "/Library/Fonts/Arial.ttf",
]

_THAI_FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/noto/NotoSansThai-Regular.ttf",
    "/usr/share/fonts/truetype/tlwg/Sarabun.ttf",
    "/Users/nurulhudaadamishaq/Library/Fonts/removed_by_sos_backup_20260430_223819/NotoSansThai-Regular.ttf",
    "/Users/nurulhudaadamishaq/Library/Fonts/NotoSansThai-Regular.ttf",
    "/Library/Fonts/NotoSansThai-Regular.ttf",
    "/System/Library/Fonts/Thonburi.ttf",
]

def _first_existing(candidates):
    for p in candidates:
        if Path(p).exists():
            return p
    return None

_latin_font_path = _first_existing(_LATIN_FONT_CANDIDATES)
_thai_font_path  = _first_existing(_THAI_FONT_CANDIDATES)

_font_cache: dict = {}

def _get_fonts(size: int):
    """คืน (font_latin, font_thai) — cache ตาม size"""
    if size not in _font_cache:
        fl = (ImageFont.truetype(_latin_font_path, size)
              if _latin_font_path else ImageFont.load_default())
        ft = (ImageFont.truetype(_thai_font_path, size)
              if _thai_font_path else fl)
        _font_cache[size] = (fl, ft)
    return _font_cache[size]


def _is_thai(ch: str) -> bool:
    return '\u0e00' <= ch <= '\u0e7f'


def _split_segments(text: str):
    """แบ่ง text เป็น list ของ (is_thai, segment_str)"""
    if not text:
        return []
    segs, cur, cur_thai = [], [], _is_thai(text[0])
    for ch in text:
        t = _is_thai(ch)
        if t == cur_thai:
            cur.append(ch)
        else:
            segs.append((cur_thai, ''.join(cur)))
            cur, cur_thai = [ch], t
    segs.append((cur_thai, ''.join(cur)))
    return segs


def _measure_mixed(text: str, size: int) -> tuple[int, int]:
    """วัดขนาด (width, height) ของ mixed Thai/Latin text"""
    font_l, font_t = _get_fonts(size)
    tmp  = Image.new("RGB", (1, 1))
    draw = ImageDraw.Draw(tmp)
    total_w, max_h = 0, 0
    for is_thai, seg in _split_segments(text):
        font = font_t if is_thai else font_l
        bb   = draw.textbbox((0, 0), seg, font=font)
        total_w += bb[2] - bb[0]
        max_h    = max(max_h, bb[3] - bb[1])
    return int(total_w), int(max_h)


def _draw_mixed(draw: ImageDraw.ImageDraw, pos: tuple, text: str,
                size: int = 32, fill: tuple = (255, 255, 255)):
    """วาด mixed Thai/Latin text โดย render แต่ละ segment ด้วย font ที่ถูกต้อง"""
    font_l, font_t = _get_fonts(size)
    x, y = pos
    for is_thai, seg in _split_segments(text):
        font = font_t if is_thai else font_l
        bb   = draw.textbbox((0, 0), seg, font=font)
        draw.text((x, y), seg, fill=fill, font=font)
        x   += bb[2] - bb[0]


def add_sos_badge(frame, event_type: str, location: str = "", time_str: str = ""):
    """
    เพิ่ม SOS event badge บนรูป (full-width, red background, Thai+English supported)

    Args:
        frame:      OpenCV image (BGR)
        event_type: "fall_warning" | "fall" | "hand_sos"
        location:   Location name (รองรับภาษาไทย)
        time_str:   Time string

    Returns:
        Modified frame with badge
    """
    h, w = frame.shape[:2]
    badge_height = 60
    font_size    = 32

    # ── Label mapping ─────────────────────────────────────────────────────────
    labels = {
        "fall_warning": "FALL WARNING",
        "fall":         "FALL DETECTED",
        "hand_sos":     "SILENT SOS HAND",
        "pose_sos":     "HANDS UP SOS",
    }
    label = labels.get(event_type, "ALERT")

    # ── Badge background (red for all alert types) ────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, badge_height), (0, 0, 200), -1)

    # ── Compose text ─────────────────────────────────────────────────────────
    parts = [label]
    if location:
        parts.append(f"@ {location}")
    if time_str:
        parts.append(time_str)
    text = "  |  ".join(parts)

    # ── Render with PIL (Thai-safe) ───────────────────────────────────────────
    try:
        pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw    = ImageDraw.Draw(pil_img)

        _, th = _measure_mixed(text, font_size)
        y_pos = max(0, (badge_height - th) // 2)

        _draw_mixed(draw, (20, y_pos), text, size=font_size, fill=(255, 255, 255))

        frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    except Exception as e:
        # Fallback: ASCII-only cv2.putText
        print(f"⚠️  PIL text rendering failed: {e}")
        fallback = f"{label} @ {location} {time_str}".strip()
        cv2.putText(frame, fallback, (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    return frame


class Visualizer:
    def __init__(self):
        self._t = []

    def fps(self, frame):
        now = time.time()
        self._t = [t for t in self._t if now - t < 1.0]
        self._t.append(now)
        cv2.putText(frame, f"FPS {len(self._t)}", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

    def banner(self, frame, atype):
        c = {"fall_warning": (0, 165, 255),
             "fall":         (0, 0, 220),
             "hand_sos":     (0, 200, 0)}
        l = {"fall_warning": "FALL WARNING",
             "fall":         "FALL DETECTED",
             "hand_sos":     "SILENT SOS HAND"}
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 50), (w, h), c, -1)
        cv2.putText(frame, f"!!! {l} !!!", (20, h - 14),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

    def hand_state(self, frame, state):
        s = ["—", "open palm", "thumb in", "SOS!"]
        c = [(150, 150, 150), (255, 200, 0), (0, 165, 255), (0, 220, 0)]
        cv2.putText(frame, f"Hand: {s[state]}", (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c[state], 1)


def frame_to_base64(frame, max_width: int = 640, quality: int = 75) -> str:
    """Convert an OpenCV frame to a compressed JPEG Base64 Data URL in-memory without saving to disk."""
    if frame is None:
        return ""
    import base64
    try:
        h, w = frame.shape[:2]
        if max_width and w > max_width:
            scale = max_width / float(w)
            frame = cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
        success, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        if not success:
            return ""
        b64 = base64.b64encode(buffer).decode('utf-8')
        return f"data:image/jpeg;base64,{b64}"
    except Exception as e:
        print(f"⚠️ frame_to_base64 error: {e}")
        return ""


def create_bbox_snapshot_base64(
    frame,
    bbox=None,
    label: str = "",
    color: tuple = (0, 0, 255),
    max_width: int = 1280,
    quality: int = 80,
    coord_size: tuple = None,
) -> str:
    """
    Creates an in-memory Base64 snapshot with bounding box only (NO BANNER).
    Does NOT write any JPEG file to disk.
    If coord_size=(coord_w, coord_h) is provided and differs from frame shape,
    bbox is automatically scaled to match the image dimensions.
    Returns: 'data:image/jpeg;base64,...'
    """
    if frame is None:
        return ""
    img = frame.copy()
    if bbox is not None:
        bx1, by1, bx2, by2 = [float(v) for v in bbox]
        ih, iw = img.shape[:2]
        if coord_size and len(coord_size) == 2:
            cw, ch = coord_size
            if cw > 0 and ch > 0 and (cw != iw or ch != ih):
                sx = iw / float(cw)
                sy = ih / float(ch)
                bx1, by1, bx2, by2 = bx1 * sx, by1 * sy, bx2 * sx, by2 * sy
        bx1, by1 = max(0, int(bx1)), max(0, int(by1))
        bx2, by2 = min(iw - 1, int(bx2)), min(ih - 1, int(by2))
        cv2.rectangle(img, (bx1, by1), (bx2, by2), color, 3)
        if label:
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            bg_y1 = max(0, by1 - lh - 10)
            bg_y2 = by1
            cv2.rectangle(img, (bx1, bg_y1), (min(iw - 1, bx1 + lw + 10), bg_y2), color, -1)
            cv2.putText(
                img,
                label,
                (bx1 + 5, bg_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
    return frame_to_base64(img, max_width=max_width, quality=quality)


import json
from datetime import datetime
from typing import Union, Optional, Tuple

def save_annotated_snapshot(
    frame,
    event_type: str,
    bbox=None,
    label: str = "",
    color: tuple = (0, 0, 255),
    coord_size: tuple = None,
    snapshot_dir: Union[str, Path] = "logs/snapshots",
    save_local: bool = True,
    extra_meta: dict = None,
    max_width: int = 1280,
    quality: int = 90,
    all_tracks: Optional[list] = None,
) -> Tuple[str, Optional[str]]:
    """
    Annotates frame with bounding box and label.
    1. Returns base64 data URL string.
    2. If save_local is True:
       - Saves JPEG image to snapshot_dir / f"{event_type}_{ts}.jpg"
       - Saves JSON metadata to f"{event_type}_{ts}.json"
       - Appends line to logs/alerts.log
       - Returns (base64_str, str(file_path))
    Otherwise returns (base64_str, None).
    """
    if frame is None:
        return "", None

    img = frame.copy()
    ih, iw = img.shape[:2]

    # Draw bounding boxes for all other people in the scene
    if all_tracks:
        for trk in all_tracks:
            t_box = trk.get("bbox")
            t_id = trk.get("track_id", trk.get("tid"))
            if t_box is None:
                continue
            # Check if this person is the primary alert person (drawn on top below)
            if bbox is not None and abs(t_box[0] - bbox[0]) < 10 and abs(t_box[1] - bbox[1]) < 10:
                continue
            tx1, ty1, tx2, ty2 = [float(v) for v in t_box]
            if coord_size and len(coord_size) == 2:
                cw, ch = coord_size
                if cw > 0 and ch > 0 and (cw != iw or ch != ih):
                    sx = iw / float(cw)
                    sy = ih / float(ch)
                    tx1, ty1, tx2, ty2 = tx1 * sx, ty1 * sy, tx2 * sx, ty2 * sy
            tx1, ty1 = max(0, int(tx1)), max(0, int(ty1))
            tx2, ty2 = min(iw - 1, int(tx2)), min(ih - 1, int(ty2))
            cv2.rectangle(img, (tx1, ty1), (tx2, ty2), (180, 180, 180), 2)
            cv2.putText(
                img,
                f"ID:{t_id}" if t_id is not None else "person",
                (tx1, max(15, ty1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

    if bbox is not None:
        bx1, by1, bx2, by2 = [float(v) for v in bbox]
        if coord_size and len(coord_size) == 2:
            cw, ch = coord_size
            if cw > 0 and ch > 0 and (cw != iw or ch != ih):
                sx = iw / float(cw)
                sy = ih / float(ch)
                bx1, by1, bx2, by2 = bx1 * sx, by1 * sy, bx2 * sx, by2 * sy
        bx1, by1 = max(0, int(bx1)), max(0, int(by1))
        bx2, by2 = min(iw - 1, int(bx2)), min(ih - 1, int(by2))
        cv2.rectangle(img, (bx1, by1), (bx2, by2), color, 3)
        if label:
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            bg_y1 = max(0, by1 - lh - 10)
            bg_y2 = by1
            cv2.rectangle(img, (bx1, bg_y1), (min(iw - 1, bx1 + lw + 10), bg_y2), color, -1)
            cv2.putText(
                img,
                label,
                (bx1 + 5, bg_y2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    b64_url = frame_to_base64(img, max_width=max_width, quality=quality)
    saved_path_str = None

    if save_local:
        try:
            s_dir = Path(snapshot_dir)
            s_dir.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            out_file = s_dir / f"{event_type}_{ts}.jpg"
            cv2.imwrite(str(out_file), img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
            saved_path_str = str(out_file)

            # Metadata JSON
            meta_file = out_file.with_suffix('.json')
            meta_data = {
                "timestamp": datetime.now().isoformat(),
                "event_type": event_type,
                "label": label,
                "image_file": str(out_file),
                "bbox": [int(v) for v in bbox] if bbox is not None else None,
                "extra": extra_meta or {}
            }
            with open(meta_file, 'w', encoding='utf-8') as f:
                json.dump(meta_data, f, indent=2, ensure_ascii=False)

            # Append to logs/alerts.log
            log_dir = Path("logs")
            log_dir.mkdir(parents=True, exist_ok=True)
            log_line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} | ALERT: {label or event_type.upper()} | file={out_file.name} | {repr(extra_meta or {})}\n"
            with open(log_dir / "alerts.log", "a", encoding="utf-8") as lf:
                lf.write(log_line)

            print(f"  [SNAPSHOT SAVED] {out_file} (logged to logs/alerts.log)")
        except Exception as e:
            print(f"⚠️  [SNAPSHOT ERROR] Failed to save snapshot: {e}")

    return b64_url, saved_path_str