import cv2, time
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

def preprocess(frame, w, h):
    if frame.shape[1]!=w or frame.shape[0]!=h:
        frame = cv2.resize(frame,(w,h))
    return frame

def _find_thai_font():
    """หาฟอนต์ที่รองรับไทย"""
    candidates = [
        "/Users/nurulhudaadamishaq/Library/Fonts/removed_by_sos_backup_20260430_223819/NotoSansThai-Regular.ttf",
        "/Library/Fonts/NotoSansThai-Regular.ttf",
        "/System/Library/Fonts/Apple Color Emoji.ttc",
    ]
    for font_path in candidates:
        if Path(font_path).exists():
            return font_path
    return None

def add_sos_badge(frame, event_type: str, location: str = "", time_str: str = ""):
    """
    เพิ่ม SOS event badge บนรูป (full-width, red background, Thai-supported)

    Args:
        frame: OpenCV image (BGR)
        event_type: "fall_warning", "fall", "hand_sos"
        location: Location name (Thai supported)
        time_str: Time string

    Returns:
        Modified frame with badge
    """
    h, w = frame.shape[:2]
    badge_height = 60

    # ─ Color mapping ─
    colors = {
        "fall_warning": (0, 165, 255),    # Orange
        "fall": (0, 0, 220),              # Red
        "hand_sos": (0, 200, 0),          # Green
    }

    # Label mapping (Thai)
    labels = {
        "fall_warning": "⚠️ คำเตือนการล้ม",
        "fall": "🚨 ตรวจจับการล้ม",
        "hand_sos": "🆘 สัญญาณช่วยเหลือ",
    }

    label = labels.get(event_type, "ALERT")
    color = colors.get(event_type, (0, 0, 220))

    # Red background for badge
    badge_color = (0, 0, 255)  # BGR: Red

    # Add red rectangle at top (full width)
    cv2.rectangle(frame, (0, 0), (w, badge_height), badge_color, -1)

    try:
        # Try using PIL for Thai text support
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil_image)

        # Find Thai font
        font_path = _find_thai_font()
        font = None
        if font_path:
            try:
                font = ImageFont.truetype(font_path, 32)
            except Exception:
                pass

        # Default font if Thai font not found
        if font is None:
            font = ImageFont.load_default()

        # Prepare text
        text_parts = []
        if label:
            text_parts.append(label)
        if location:
            text_parts.append(f"@ {location}")
        if time_str:
            text_parts.append(time_str)

        text = " | ".join(text_parts)

        # Draw white text on red badge
        text_color = (255, 255, 255)  # White
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        # Center text vertically
        y_pos = (badge_height - text_height) // 2
        x_pos = 20  # Padding from left

        draw.text((x_pos, y_pos), text, fill=text_color, font=font)

        # Convert back to OpenCV format
        frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    except Exception as e:
        # Fallback: use cv2.putText if PIL fails
        print(f"⚠️ PIL text rendering failed: {e}, using fallback")
        text = f"{label} @ {location} {time_str}".strip()
        cv2.putText(
            frame, text, (20, 45),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
        )

    return frame

class Visualizer:
    def __init__(self):
        self._t = []

    def fps(self, frame):
        now = time.time()
        self._t = [t for t in self._t if now-t<1.0]
        self._t.append(now)
        cv2.putText(frame,f"FPS {len(self._t)}",(10,24),
                    cv2.FONT_HERSHEY_SIMPLEX,0.65,(200,200,200),1)

    def banner(self, frame, atype):
        c = {"fall_warning":(0,165,255),"fall":(0,0,220),"hand_sos":(0,200,0)}.get(atype,(0,0,220))
        l = {"fall_warning":"FALL WARNING","fall":"FALL DETECTED","hand_sos":"SILENT SOS HAND"}.get(atype,"ALERT")
        h,w = frame.shape[:2]
        cv2.rectangle(frame,(0,h-50),(w,h),c,-1)
        cv2.putText(frame,f"!!! {l} !!!",(20,h-14),
                    cv2.FONT_HERSHEY_DUPLEX,1.0,(255,255,255),2)

    def hand_state(self, frame, state):
        s = ["—","open palm","thumb in","SOS!"]
        c = [(150,150,150),(255,200,0),(0,165,255),(0,220,0)]
        cv2.putText(frame,f"Hand: {s[state]}",(10,56),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,c[state],1)

