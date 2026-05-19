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
    return total_w, max_h


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
             "hand_sos":     (0, 200, 0)}.get(atype, (0, 0, 220))
        l = {"fall_warning": "FALL WARNING",
             "fall":         "FALL DETECTED",
             "hand_sos":     "SILENT SOS HAND"}.get(atype, "ALERT")
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, h - 50), (w, h), c, -1)
        cv2.putText(frame, f"!!! {l} !!!", (20, h - 14),
                    cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2)

    def hand_state(self, frame, state):
        s = ["—", "open palm", "thumb in", "SOS!"]
        c = [(150, 150, 150), (255, 200, 0), (0, 165, 255), (0, 220, 0)]
        cv2.putText(frame, f"Hand: {s[state]}", (10, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, c[state], 1)