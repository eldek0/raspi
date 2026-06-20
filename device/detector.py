import logging
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from camera import CameraModule
from config import (
    DEVICE_ID, PENDING_DIR,
    YOLO_MODEL, YOLO_CONF, IOU_THRESHOLD, IMG_SIZE,
    DETECTION_INTERVAL, ALERT_DURATION_SECS, ALERT_REPEAT_INTERVAL, NORMAL_EVENT_INTERVAL,
    DISPLAY,
)
from store import EventStore

logger = logging.getLogger(__name__)
logger_temp = logging.getLogger('temperatura')
logger_cam  = logging.getLogger('camara')

_NO_HELMET_LABEL = 'NO-Hardhat'
_W1_DEV_PATH     = '/sys/bus/w1/devices/'
_W1_PATTERN      = re.compile(r'^[0-9a-fA-F]+-[0-9a-fA-F]+$')
_w1_device_path: Optional[str] = None  # cached after first successful scan


def _find_w1_device() -> Optional[str]:
    global _w1_device_path
    if _w1_device_path and os.path.exists(_w1_device_path):
        return _w1_device_path
    try:
        for name in os.listdir(_W1_DEV_PATH):
            if _W1_PATTERN.match(name):
                path = os.path.join(_W1_DEV_PATH, name, 'w1_slave')
                if os.path.exists(path):
                    _w1_device_path = path
                    logger_temp.info(f'Sensor 1-Wire encontrado: {path}')
                    return path
    except OSError:
        pass
    logger_temp.warning(f'Sensor 1-Wire no encontrado en {_W1_DEV_PATH}')
    return None


def read_temperature() -> float:
    path = _find_w1_device()
    if path is None:
        return -1.0
    try:
        with open(path, 'r') as f:
            data = f.read()
        if 't=' not in data:
            return -1.0
        return float(data.split('t=')[1].strip()) / 1000.0
    except (OSError, ValueError):
        return -1.0


class Detector:
    def __init__(self, camera: CameraModule, store: EventStore):
        self._camera = camera
        self._store  = store

        from ultralytics import YOLO
        model_path = self._resolve_model(YOLO_MODEL)
        logger.info('Cargando modelo YOLO en memoria...')
        self._model = YOLO(model_path)
        logger.info('Modelo YOLO listo')

        self._violation_since: Optional[float] = None
        self._last_alert_time: Optional[float] = None
        self._clear_counter                    = 0
        self._last_detection: Optional[tuple[int, int]] = None
        self._helmet_on: bool = True
        self._last_boxes: list = []
        self._last_temp: float = -1.0

    def get_last_detection(self) -> dict:
        return {
            'is_alert':    not self._helmet_on,
            'detection_x': self._last_detection[0] if self._last_detection else None,
            'detection_y': self._last_detection[1] if self._last_detection else None,
        }

    @staticmethod
    def _resolve_model(model_spec: str) -> str:
        if os.path.exists(model_spec):
            logger.info(f'Modelo local: {model_spec}')
            return model_spec

        from config import DATA_DIR
        cache_dir  = DATA_DIR / 'models'
        # slug: "owner/repo-name" → "owner__repo-name.pt"
        cache_name = model_spec.replace('/', '__') + '.pt'
        cache_path = cache_dir / cache_name

        if cache_path.exists():
            logger.info(f'Modelo en caché: {cache_path}')
            return str(cache_path)

        logger.info(f'Descargando modelo HuggingFace {model_spec} → {cache_path} ...')
        from huggingface_hub import hf_hub_download
        downloaded = hf_hub_download(repo_id=model_spec, filename='best.pt')
        cache_dir.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy2(downloaded, cache_path)
        logger.info(f'Modelo guardado: {cache_path}')
        return str(cache_path)

    def run(self) -> None:
        last_normal = 0.0

        try:
            while True:
                now   = time.time()
                frame = self._camera.latest_frame()

                if frame is not None:
                    detection        = self._infer(frame)
                    helmet           = detection is None  # True = casco puesto / sin infracción
                    self._last_detection = detection
                    self._helmet_on      = helmet

                    if DISPLAY:
                        self._camera.set_annotated_frame(self._annotate(frame, helmet, now))

                    if not helmet:
                        self._clear_counter = 0
                        if self._violation_since is None:
                            self._violation_since = now
                            logger_cam.warning('Sin casco detectado — iniciando conteo')
                        elapsed         = now - self._violation_since
                        time_since_last = (now - self._last_alert_time) if self._last_alert_time is not None else float('inf')
                        if elapsed >= ALERT_DURATION_SECS and time_since_last >= ALERT_REPEAT_INTERVAL:
                            logger_cam.warning(
                                f'Sin casco por {elapsed:.0f}s — disparando alerta '
                                f'(repeat cada {ALERT_REPEAT_INTERVAL:.0f}s)'
                            )
                            self._trigger_alert(detection)
                            self._last_alert_time = now
                            last_normal = now
                        elif self._last_alert_time is None:
                            logger_cam.debug(f'Sin casco: {elapsed:.0f}s / {ALERT_DURATION_SECS}s')
                    else:
                        self._clear_counter += 1
                        if self._clear_counter >= 3:
                            if self._violation_since is not None:
                                logger_cam.info('Casco detectado — reseteando conteo')
                            self._violation_since = None
                            self._last_alert_time = None

                    if now - last_normal >= NORMAL_EVENT_INTERVAL:
                        last_normal = now
                        if helmet:
                            logger_temp.info('Evento normal: casco puesto')
                            self._enqueue_normal_event(True, None)

                time.sleep(DETECTION_INTERVAL)

        except KeyboardInterrupt:
            pass

    def _infer(self, frame: np.ndarray) -> Optional[tuple[int, int]]:
        results = self._model(frame, conf=YOLO_CONF, iou=IOU_THRESHOLD, imgsz=IMG_SIZE, verbose=False)
        best_conf = -1.0
        best_xy: Optional[tuple[int, int]] = None
        boxes = []
        for r in results:
            for box in r.boxes:
                label = r.names[int(box.cls)]
                conf  = float(box.conf)
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                boxes.append((label, conf, x1, y1, x2, y2))
                if label == _NO_HELMET_LABEL and conf >= YOLO_CONF and conf > best_conf:
                    best_conf = conf
                    best_xy = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        self._last_boxes = boxes
        if best_xy is not None:
            logger.debug(f'Detección sin casco: conf={best_conf:.2f} xy={best_xy}')
        return best_xy

    def _trigger_alert(self, detection: tuple[int, int]) -> None:
        event_id  = str(uuid.uuid4())
        partition = datetime.now(timezone.utc).strftime('%Y/%m/%d/%H')

        logger_cam.warning(f'Alerta enviada: event_id={event_id} xy={detection}')
        temperature = read_temperature()
        self._last_temp = temperature
        photo_path = self._camera.sacar_foto(PENDING_DIR)
        video_path = self._camera.grabar_clip(PENDING_DIR, seconds=ALERT_DURATION_SECS)

        payload = {
            'event_id':    event_id,
            'type':        'alert',
            'device_id':   DEVICE_ID,
            'is_alert':    True,
            'helmet':      False,
            'detection_x': detection[0],
            'detection_y': detection[1],
            'temperature': temperature,
            'timestamp':   datetime.now(timezone.utc).isoformat(),
        }
        self._store.enqueue(
            'photo', event_id=event_id, data=payload,
            filepath=photo_path, filename=f'{partition}/{event_id}.jpg',
        )
        self._store.enqueue(
            'video', event_id=event_id,
            filepath=video_path, filename=f'{partition}/{event_id}.mp4',
        )
        logger_cam.warning(f'Alerta encolada: photo={photo_path} video={video_path}')

    def _annotate(self, frame: np.ndarray, helmet: bool, now: float) -> bytes:
        disp = frame.copy()
        H, W = disp.shape[:2]

        # Bounding boxes
        for label, conf, x1, y1, x2, y2 in self._last_boxes:
            no_helmet_box = label == _NO_HELMET_LABEL
            color = (0, 0, 220) if no_helmet_box else (0, 200, 60)
            text  = f'{"SIN CASCO" if no_helmet_box else "CASCO"}  {conf:.0%}'
            cv2.rectangle(disp, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(disp, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
            cv2.putText(disp, text, (x1 + 3, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        # Status bar
        bar_color = (0, 0, 160) if not helmet else (0, 130, 0)
        cv2.rectangle(disp, (0, 0), (W, 38), bar_color, -1)

        if not helmet and self._violation_since is not None:
            elapsed      = now - self._violation_since
            status_label = f'SIN CASCO  {elapsed:.0f}s'
        elif helmet:
            status_label = 'CASCO DETECTADO'
        else:
            status_label = 'MONITOREANDO'

        cv2.putText(disp, status_label, (8, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)

        if self._last_temp > 0:
            temp_str = f'{self._last_temp:.1f} C'
            (tw, _), _ = cv2.getTextSize(temp_str, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.putText(disp, temp_str, (W - tw - 8, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

        _, jpg = cv2.imencode('.jpg', disp, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return jpg.tobytes()

    def _enqueue_normal_event(self, helmet: bool, detection: Optional[tuple[int, int]]) -> None:
        event_id    = str(uuid.uuid4())
        temperature = read_temperature()
        self._last_temp = temperature
        logger_temp.info(f'Temperatura leída: {temperature}°C')
        payload  = {
            'event_id':      event_id,
            'type':          'normal',
            'device_id':     DEVICE_ID,
            'is_alert':      False,
            'helmet':        helmet,
            'detection_x':   detection[0] if detection else None,
            'detection_y':   detection[1] if detection else None,
            'temperature':   temperature,
            'timestamp':     datetime.now(timezone.utc).isoformat(),
        }
        self._store.enqueue('event', event_id=event_id, data=payload)
