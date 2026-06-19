"""
Construction helmet detection — video file or webcam.

Usage:
    python detect_video.py                   # webcam
    python detect_video.py path/to/video.mp4

Set PROCESS_EVERY_N_FRAMES > 1 for low-FPS / embedded hardware.
After training on SHWD, update MODEL_PATH to models/shwd_best.pt.
"""

import sys
import cv2
import numpy as np
from ultralytics import YOLO
import time
from pathlib import Path

# ─────────────────────────── CONFIGURATION ────────────────────────────

VIDEO_SOURCE = sys.argv[1] if len(sys.argv) > 1 else 0

# Best available model — update to models/shwd_best.pt after SHWD training
MODEL_PATH = "models/helmet_model_medium.pt"
PERSON_MODEL_PATH = "yolov8n.pt"    # fallback person detector

CONFIDENCE         = 0.20   # lower threshold catches more real detections; raise if too many false positives
IOU_THRESHOLD      = 0.45
IMG_SIZE           = 640

# 1 = every frame; 3 = ~10 fps on a 30 fps feed; 5 = very low FPS
PROCESS_EVERY_N_FRAMES = 3

USE_PERSON_FALLBACK = True   # detect uncovered people with yolov8n
USE_COLOR_FALLBACK  = True   # check head-region color if no model coverage

SAVE_OUTPUT  = True
OUTPUT_PATH  = "output_detection.mp4"
DISPLAY      = True          # False for headless / SSH

# BGR colors
COLOR_HELMET   = (0, 220,   0)
COLOR_NO_HELMET= (0,   0, 220)
COLOR_WARN     = (0, 165, 255)

# ──────────────────────────── CLASS MAPPING ───────────────────────────

def build_class_mapper(model_names: dict) -> dict:
    """
    Map class_id → 'con_casco' | 'sin_casco' | None for any supported model:
      - keremberke / Hardhat format : {0:'Hardhat',  1:'NO-Hardhat'}
      - SHWD format                 : {0:'hat',      1:'person'}     (person = bare head)
      - generic variants            : helmet, safety_helmet, head, …
    """
    has_helmet_cls = any(
        'hat' in n.lower() or 'helmet' in n.lower()
        for n in model_names.values()
    )

    mapping = {}
    for idx, name in model_names.items():
        low = name.lower().strip()

        # ── no-helmet patterns ──────────────────────────────────────
        if low in ('no-hardhat', 'no_hardhat', 'no-helmet', 'no_helmet',
                   'nohardhat', 'without_helmet', 'head', 'bare_head'):
            mapping[idx] = 'sin_casco'
        elif low.startswith('no-') or low.startswith('no_'):
            mapping[idx] = 'sin_casco'
        elif 'without' in low:
            mapping[idx] = 'sin_casco'
        # 'person' in a SHWD-style model means "bare head / no helmet visible"
        elif low == 'person' and has_helmet_cls:
            mapping[idx] = 'sin_casco'

        # ── with-helmet patterns ─────────────────────────────────────
        elif 'hat' in low or 'helmet' in low:
            mapping[idx] = 'con_casco'

        else:
            mapping[idx] = None   # ignore

    return mapping


# ───────────────────────────── DETECTION ──────────────────────────────

def run_helmet_model(model, frame, class_mapper):
    """Return list of (x1,y1,x2,y2, label, conf)."""
    results = model(frame, conf=CONFIDENCE, iou=IOU_THRESHOLD,
                    imgsz=IMG_SIZE, verbose=False)
    dets = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            cls   = int(box.cls[0])
            label = class_mapper.get(cls)
            if label is None:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            dets.append((x1, y1, x2, y2, label, conf))
    return dets


def run_person_model(model, frame):
    """Return list of (x1,y1,x2,y2) for each detected person."""
    results = model(frame, conf=0.35, iou=IOU_THRESHOLD,
                    imgsz=IMG_SIZE, classes=[0], verbose=False)
    people = []
    if results and results[0].boxes is not None:
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            people.append((x1, y1, x2, y2))
    return people


def _iou(a, b):
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    if ix1 >= ix2 or iy1 >= iy2:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    aa = (a[2]-a[0])*(a[3]-a[1]); ab = (b[2]-b[0])*(b[3]-b[1])
    return inter / (aa + ab - inter)


def _classify_person(person, helmet_dets, frame):
    """Return (label, conf) for a person box using helmet-model hits + colour fallback."""
    px1, py1, px2, py2 = person
    person_h = py2 - py1
    person_w = px2 - px1

    # Head zone: top 30% of the person bounding box
    head_bottom = py1 + person_h * 0.30

    best = {'con_casco': None, 'sin_casco': None}
    for (hx1, hy1, hx2, hy2, label, conf) in helmet_dets:
        helmet_cy = (hy1 + hy2) / 2
        helmet_cx = (hx1 + hx2) / 2

        on_head = (
            hy2 >= py1 - person_h * 0.10 and            # helmet bottom at or near top of person
            hy2 <= py1 + person_h * 0.45 and            # helmet bottom not too low
            helmet_cy >= py1 - person_h * 0.15 and      # center can be slightly above person top
            helmet_cy < head_bottom and                  # center within head zone
            helmet_cx >= px1 - person_w * 0.10 and      # small horizontal tolerance
            helmet_cx <= px2 + person_w * 0.10
        )
        if on_head:
            if best[label] is None or conf > best[label]:
                best[label] = conf
    if best['con_casco'] is not None and best['con_casco'] >= 0.65:
        return 'con_casco', best['con_casco']
    if best['sin_casco'] is not None:
        return 'sin_casco', best['sin_casco']
    if USE_COLOR_FALLBACK:
        ratio = _head_color_ratio(frame, person)
        if ratio > 0.22:
            color_conf = round(0.35 + ratio * 0.4, 2)
            if color_conf >= 0.65:
                return 'con_casco', color_conf
    # Only flag sin_casco with low confidence when we have no other signal
    return 'sin_casco', 0.30


def _head_color_ratio(frame, person):
    """Return fraction of hard-hat-colored pixels in the top 25% of a person box."""
    px1, py1, px2, py2 = person
    h = py2 - py1; w = px2 - px1
    hx1 = max(0, px1 + int(w * 0.15))
    hx2 = min(frame.shape[1], px2 - int(w * 0.15))
    hy1 = max(0, py1)
    hy2 = min(frame.shape[0], py1 + int(h * 0.25))
    if hx2 <= hx1 or hy2 <= hy1:
        return 0.0

    hsv = cv2.cvtColor(frame[hy1:hy2, hx1:hx2], cv2.COLOR_BGR2HSV)
    masks = [
        cv2.inRange(hsv, np.array([15,  80,  80]), np.array([45,  255, 255])),  # yellow
        cv2.inRange(hsv, np.array([ 5, 100, 100]), np.array([25,  255, 255])),  # orange
        cv2.inRange(hsv, np.array([ 0,   0, 180]), np.array([180,  50, 255])),  # white
        cv2.inRange(hsv, np.array([90,  80,  80]), np.array([130, 255, 255])),  # blue
        cv2.inRange(hsv, np.array([35,  80,  80]), np.array([85,  255, 255])),  # green
        cv2.inRange(hsv, np.array([ 0, 100, 100]), np.array([10,  255, 255]))   # red lo
        | cv2.inRange(hsv, np.array([160,100,100]), np.array([180, 255, 255])), # red hi
    ]
    combined = masks[0].copy()
    for m in masks[1:]:
        combined |= m
    total = combined.size
    return cv2.countNonZero(combined) / total if total else 0.0


def merge_detections(helmet_dets, person_boxes, frame):
    """Person-centric: one output box per detected person, classified by helmet presence.
    Standalone helmet detections with no matching person are discarded."""
    if not person_boxes:
        return []
    return [(*pb, *_classify_person(pb, helmet_dets, frame)) for pb in person_boxes]


# ─────────────────────────────── HUD ──────────────────────────────────

def draw_detections(frame, dets):
    stats = {'con_casco': 0, 'sin_casco': 0}
    for (x1, y1, x2, y2, label, conf) in dets:
        if label == 'con_casco':
            color = COLOR_HELMET; text = f"CASCO {conf:.0%}"; thick = 2
            stats['con_casco'] += 1
        else:
            color = COLOR_NO_HELMET; text = f"SIN CASCO {conf:.0%}"; thick = 3
            stats['sin_casco'] += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thick)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = max(y1, th + 8)
        cv2.rectangle(frame, (x1, ty - th - 6), (x1 + tw + 8, ty + 2), color, -1)
        cv2.putText(frame, text, (x1 + 4, ty - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return frame, stats


def draw_hud(frame, stats, fps, frame_idx):
    h, w = frame.shape[:2]
    total = stats['con_casco'] + stats['sin_casco']
    pct   = (stats['con_casco'] / total * 100) if total else 0

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 62), (15, 15, 28), -1)
    frame = cv2.addWeighted(overlay, 0.78, frame, 0.22, 0)

    cv2.putText(frame, "DETECCION DE CASCOS", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 255), 2)
    cv2.putText(frame, f"FPS:{fps:.0f}  #{frame_idx}", (w - 175, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (140, 140, 140), 1)

    bx, by, bw = 10, 36, 240
    cv2.rectangle(frame, (bx, by), (bx + bw, by + 16), (45, 45, 58), -1)
    fill = int(bw * pct / 100)
    bar_color = COLOR_HELMET if pct >= 80 else COLOR_WARN if pct >= 50 else COLOR_NO_HELMET
    if fill > 0:
        cv2.rectangle(frame, (bx, by), (bx + fill, by + 16), bar_color, -1)
    cv2.putText(frame, f"Cumplimiento {pct:.0f}%  ({stats['con_casco']}/{total})",
                (bx + 3, by + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    if stats['sin_casco'] > 0:
        alert = f"! {stats['sin_casco']} SIN CASCO"
        (aw, _), _ = cv2.getTextSize(alert, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
        cv2.putText(frame, alert, (w - aw - 12, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, COLOR_NO_HELMET, 2)

    return frame


# ───────────────────────────── MAIN LOOP ──────────────────────────────

def run():
    # ── Load models ────────────────────────────────────────────────────
    if not Path(MODEL_PATH).exists():
        candidates = list(Path("models").glob("*.pt"))
        print(f"ERROR: {MODEL_PATH} not found.")
        if candidates:
            print("Available models:")
            for p in candidates:
                print(f"  {p}")
        return

    helmet_model = YOLO(MODEL_PATH)
    class_mapper = build_class_mapper(helmet_model.names)
    print(f"Helmet model : {MODEL_PATH}")
    print(f"  classes    : {helmet_model.names}")
    print(f"  mapping    : {class_mapper}")

    person_model = None
    if USE_PERSON_FALLBACK:
        try:
            person_model = YOLO(PERSON_MODEL_PATH)
            print(f"Person model : {PERSON_MODEL_PATH}")
        except Exception:
            print(f"Warning: could not load person model {PERSON_MODEL_PATH} — fallback disabled")

    # ── Open video source ───────────────────────────────────────────────
    src = int(VIDEO_SOURCE) if VIDEO_SOURCE == 0 or str(VIDEO_SOURCE) == '0' else VIDEO_SOURCE
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        print(f"ERROR: cannot open {VIDEO_SOURCE}")
        return

    src_fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_fr   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    is_file    = not (VIDEO_SOURCE == 0 or str(VIDEO_SOURCE) == '0')

    print(f"\nSource: {VIDEO_SOURCE}  {width}x{height} @ {src_fps:.0f}fps"
          + (f"  ({total_fr} frames)" if is_file else ""))
    print(f"Processing every {PROCESS_EVERY_N_FRAMES} frame(s)  "
          f"→ effective ~{src_fps / PROCESS_EVERY_N_FRAMES:.0f} detections/s")
    print("Controls: q=quit  p=pause  s=screenshot\n")

    writer = None
    if SAVE_OUTPUT:
        out_fps = max(1, src_fps / PROCESS_EVERY_N_FRAMES)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, out_fps, (width, height))

    frame_idx   = 0
    t_start     = time.time()
    last_dets   = []
    last_stats  = {'con_casco': 0, 'sin_casco': 0}
    paused      = False

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

        # Run inference only every N frames; reuse last result on skipped frames
        if frame_idx % PROCESS_EVERY_N_FRAMES == 0 or frame_idx == 1:
            helmet_dets  = run_helmet_model(helmet_model, frame, class_mapper)
            person_boxes = run_person_model(person_model, frame) if person_model else []
            last_dets    = merge_detections(helmet_dets, person_boxes, frame)

        display = frame.copy()
        display, last_stats = draw_detections(display, last_dets)

        elapsed = max(time.time() - t_start, 1e-6)
        fps     = frame_idx / elapsed
        display = draw_hud(display, last_stats, fps, frame_idx)

        if writer:
            writer.write(display)

        if DISPLAY:
            cv2.imshow("Helmet Detection  [q=quit  p=pause  s=save]", display)
            key = cv2.waitKey(0 if paused else 1) & 0xFF
            if key == ord('q'):
                print("Stopped by user.")
                break
            elif key == ord('p'):
                paused = not paused
                print("Paused." if paused else "Resumed.")
            elif key == ord('s'):
                fname = f"screenshot_{frame_idx:06d}.jpg"
                cv2.imwrite(fname, display)
                print(f"Saved {fname}")

        if is_file and total_fr > 0 and frame_idx % 300 == 0:
            pct = frame_idx / total_fr * 100
            print(f"  {pct:.0f}%  frame {frame_idx}/{total_fr}  "
                  f"fps={fps:.1f}  "
                  f"con={last_stats['con_casco']}  sin={last_stats['sin_casco']}")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"\nFinished. {frame_idx} frames in {elapsed:.1f}s "
          f"({frame_idx/elapsed:.1f} fps average)")
    if SAVE_OUTPUT and writer:
        print(f"Output saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    run()
