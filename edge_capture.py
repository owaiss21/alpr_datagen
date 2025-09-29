# edge_capture.py
import argparse
import os
import sys
import time
import signal
import threading
from pathlib import Path
from collections import deque, defaultdict
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
import imagehash
from ultralytics import YOLO


# ---------------------------
# Constants & simple mappings
# ---------------------------
COCO_CLASS_MAP = {
    "person": 0,
    "car": 2,
}
DEFAULT_WEIGHTS = "yolov8n.pt"

STOP_EVENT = threading.Event()


# ---------------------------
# Utils
# ---------------------------
def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def timestamp_slug():
    now = datetime.now()
    return now.strftime("%Y-%m-%d_%H-%M-%S_") + f"{int(now.microsecond/1000):03d}"

def build_outdir(base: Path):
    # CHANGED: single daily folder YYYY_MM_DD (was nested YYYY/MM/DD)
    now = datetime.now()
    day = f"{now.year:04d}_{now.month:02d}_{now.day:02d}"
    outdir = base / day
    safe_mkdir(outdir)
    return outdir

def pil_hash(img_bgr: np.ndarray):
    small = cv2.resize(img_bgr, (256, 256), interpolation=cv2.INTER_AREA)
    img_rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    return imagehash.dhash(pil_img)

def is_stream_url(s: str) -> bool:
    s = s.lower()
    return s.startswith("rtsp://") or s.startswith("http://") or s.startswith("https://")

def normalize_source(src: str):
    """
    Accepts: int camera index (e.g., "0"), a stream URL, or a path to a video file.
    We convert to int only if it's purely digits AND not a real file.
    """
    if isinstance(src, int):
        return src
    if src.isdigit() and not Path(src).exists():
        return int(src)
    return src


# ---------------------------
# RateLimiter / Cooldown / DuplicateSuppressor
# ---------------------------
class RateLimiter:
    def __init__(self, max_per_min=30, window_secs=60):
        self.max_per_min = int(max_per_min)
        self.window_secs = int(window_secs)
        self._events = defaultdict(deque)

    def allow(self, class_id: int, now: float) -> bool:
        q = self._events[class_id]
        cutoff = now - self.window_secs
        while q and q[0] < cutoff:
            q.popleft()
        if len(q) < self.max_per_min:
            q.append(now)
            return True
        return False


class Cooldown:
    def __init__(self, cooldown_s=2.0):
        self.cooldown_s = float(cooldown_s)
        self._next_ok_at = defaultdict(float)

    def allow(self, class_id: int, now: float) -> bool:
        if now >= self._next_ok_at[class_id]:
            self._next_ok_at[class_id] = now + self.cooldown_s
            return True
        return False


class DuplicateSuppressor:
    def __init__(self, max_keep=200, hamming_thresh=6):
        self._recent = deque(maxlen=int(max_keep))
        self.hamming_thresh = int(hamming_thresh)

    def allow(self, img_bgr: np.ndarray) -> bool:
        try:
            h = pil_hash(img_bgr)
        except Exception:
            return True
        for prev in self._recent:
            if h - prev <= self.hamming_thresh:
                return False
        self._recent.append(h)
        return True


# ---------------------------
# Detection Pipeline
# ---------------------------
class EdgeSampler:
    def __init__(
        self,
        mode: str,
        source,
        out_base: Path,
        weights: str = DEFAULT_WEIGHTS,
        imgsz: int = 640,
        conf: float = 0.35,
        iou: float = 0.5,
        save_crops: bool = False,
        cooldown_s: float = 2.0,
        max_per_min: int = 30,
        use_hash: bool = True,
        hash_hamming_thresh: int = 6,
        show: bool = False,
        verbose: bool = False,  # ADDED: compact per-frame logging
    ):
        self.mode = mode
        self.source = source
        self.out_base = out_base
        self.weights = weights
        self.imgsz = int(imgsz)
        self.conf = float(conf)
        self.iou = float(iou)
        self.save_crops = bool(save_crops)
        self.show = bool(show)
        self.verbose = bool(verbose)  # ADDED

        if mode == "cars":
            self.allowed_cls = {COCO_CLASS_MAP["car"]}
            self.tag = "car"
        elif mode == "people":
            self.allowed_cls = {COCO_CLASS_MAP["person"]}
            self.tag = "person"
        else:
            raise ValueError("mode must be one of: cars, people")

        self.ratelimiter = RateLimiter(max_per_min=max_per_min, window_secs=60)
        self.cooldown = Cooldown(cooldown_s=cooldown_s)
        self.dupe = DuplicateSuppressor(max_keep=200, hamming_thresh=hash_hamming_thresh) if use_hash else None

        # Model (Ultralytics decides device internally; no torch import)
        self.model = YOLO(self.weights)

        # Video source
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open source: {self.source}")

        self.font = cv2.FONT_HERSHEY_SIMPLEX

        # Detect if input is a file so we stop at EOF
        self._is_file = isinstance(self.source, str) and Path(self.source).exists() and not is_stream_url(self.source)

    def _should_save(self, class_id: int, frame_bgr: np.ndarray) -> bool:
        now = time.monotonic()
        if not self.cooldown.allow(class_id, now):
            return False
        if not self.ratelimiter.allow(class_id, now):
            return False
        if self.dupe is not None and not self.dupe.allow(frame_bgr):
            return False
        return True

    def _draw_overlay(self, frame, fps=None):
        y = 24
        cv2.putText(frame, f"Mode: {self.mode}", (10, y), self.font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
        y += 22
        if fps is not None:
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, y), self.font, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

    def _save_sample(self, frame_bgr, xyxy=None):
        outdir = build_outdir(self.out_base)
        slug = timestamp_slug()
        if self.save_crops and xyxy is not None:
            x1, y1, x2, y2 = map(int, xyxy)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = max(x1 + 1, x2), max(y1 + 1, y2)
            crop = frame_bgr[y1:y2, x1:x2]
            path = outdir / f"{slug}_{self.tag}_crop.jpg"
            cv2.imwrite(str(path), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])
            return path  # CHANGED: return path for logging
        else:
            path = outdir / f"{slug}_{self.tag}.jpg"
            cv2.imwrite(str(path), frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
            return path  # CHANGED: return path for logging

    def run(self):
        fps_smooth = None
        t_prev = time.perf_counter()

        try:
            while not STOP_EVENT.is_set():
                ok, frame = self.cap.read()
                if not ok:
                    if self._is_file:
                        # End of video file — stop cleanly
                        break
                    # For streams/cams: brief backoff + reconnect
                    time.sleep(0.2)
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.source)
                    continue

                # Inference (keep Ultralytics quiet; we print our own concise line)
                t0 = time.perf_counter()
                results = self.model.predict(
                    source=frame,
                    imgsz=self.imgsz,
                    conf=self.conf,
                    iou=self.iou,
                    verbose=False,  # CHANGED: no extra Ultralytics logs
                )
                infer_ms = (time.perf_counter() - t0) * 1000.0

                checked = 0
                any_saved = False
                saved_path = None

                for r in results:
                    boxes = r.boxes
                    if boxes is None or len(boxes) == 0:
                        continue
                    for b in boxes:
                        cls_id = int(b.cls.item())
                        score = float(b.conf.item())
                        if cls_id in self.allowed_cls and score >= self.conf:
                            checked += 1
                            if self._should_save(cls_id, frame):
                                saved_path = self._save_sample(
                                    frame, xyxy=b.xyxy[0].tolist() if self.save_crops else None
                                )
                                any_saved = True
                                break

                # ADDED: one concise line per frame when --verbose
                if self.verbose:
                    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    sp = str(saved_path) if saved_path else "-"
                    print(f"{ts} | {infer_ms:.1f} ms | det={checked} | saved={sp}")

                if self.show:
                    fps = 1.0 / max(1e-6, (time.perf_counter() - t_prev))
                    fps_smooth = fps if fps_smooth is None else (0.9 * fps_smooth + 0.1 * fps)
                    t_prev = time.perf_counter()
                    self._draw_overlay(frame, fps=fps_smooth)
                    cv2.imshow("edge_capture", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

                if any_saved:
                    time.sleep(0.01)

        finally:
            self.cap.release()
            if self.show:
                cv2.destroyAllWindows()


# ---------------------------
# Signals
# ---------------------------
def _on_sigint(sig, frame):
    STOP_EVENT.set()

signal.signal(signal.SIGINT, _on_sigint)
if hasattr(signal, "SIGTERM"):
    signal.signal(signal.SIGTERM, _on_sigint)


# ---------------------------
# CLI
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Edge data-gatherer for cars / people with cool-down, caps, and dedup (ultralytics-only)."
    )
    p.add_argument("--mode", choices=["cars", "people"], required=True)
    p.add_argument("--source", required=True,
                   help="Camera index (e.g., 0), RTSP/HTTP URL, or path to a video file.")
    p.add_argument("--out", type=Path, required=True, help="Output base directory.")
    p.add_argument("--weights", default=DEFAULT_WEIGHTS, help="YOLOv8 weights (.pt).")
    p.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    p.add_argument("--conf", type=float, default=0.35, help="Confidence threshold.")
    p.add_argument("--iou", type=float, default=0.5, help="IOU threshold.")
    p.add_argument("--save-crops", action="store_true", help="Save cropped detections instead of full frames.")
    p.add_argument("--cooldown-s", type=float, default=2.0, help="Per-class cooldown seconds after a save.")
    p.add_argument("--max-per-min", type=int, default=30, help="Per-class max saves per minute.")
    p.add_argument("--no-hash", action="store_true", help="Disable near-duplicate suppression.")
    p.add_argument("--hash-hamming-thresh", type=int, default=6, help="Lower = stricter duplicate filtering.")
    p.add_argument("--show", action="store_true", help="Preview window (desktop testing).")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-frame timing, detections considered, and save path (no extra Ultralytics logs).")
    return p.parse_args()


def main():
    args = parse_args()
    src = normalize_source(args.source)
    sampler = EdgeSampler(
        mode=args.mode,
        source=src,
        out_base=args.out,
        weights=args.weights,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        save_crops=args.save_crops,
        cooldown_s=args.cooldown_s,
        max_per_min=args.max_per_min,
        use_hash=not args.no_hash,
        hash_hamming_thresh=args.hash_hamming_thresh,
        show=args.show,
        verbose=args.verbose,
    )
    sampler.run()


if __name__ == "__main__":
    main()
