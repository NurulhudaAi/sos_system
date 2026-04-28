import subprocess
import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox


REPO_ROOT = Path(__file__).resolve().parent
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
MAIN = REPO_ROOT / "main.py"


def start_detector(video_path: str) -> None:
    if not PYTHON.exists():
        messagebox.showerror("SOS Detector", f"Python not found: {PYTHON}")
        return
    if not MAIN.exists():
        messagebox.showerror("SOS Detector", f"main.py not found: {MAIN}")
        return

    subprocess.Popen(
        [str(PYTHON), str(MAIN), video_path],
        cwd=str(REPO_ROOT),
    )


def choose_file() -> None:
    root = tk.Tk()
    root.withdraw()
    root.update()

    video_path = filedialog.askopenfilename(
        title="Select a video file",
        filetypes=[
            ("Video files", "*.mp4 *.mov *.m4v *.avi *.mkv"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()

    if not video_path:
        return

    start_detector(video_path)


if __name__ == "__main__":
    choose_file()
