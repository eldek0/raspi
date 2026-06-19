"""Alert-only mode — loads YOLO, watches sensor, fires alert on sustained no-helmet detection.
No periodic normal events. No screenshot command listener."""
import logging
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Optional

import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')

logger     = logging.getLogger(__name__)
logger_cam = logging.getLogger('camara')

from camera import CameraModule, start_camera_server
from config import (
    CAMERA_DEVICE, CAMERA_BUFFER_SECS, DATA_DIR, PENDING_DIR,
    DEVICE_ID, YOLO_MODEL, YOLO_CONF,
    DETECTION_INTERVAL, ALERT_DURATION_SECS,
)
from detector import Detector, read_temperature
from store import EventStore
from worker import queue_worker


class AlertOnlyDetector:
    """Detector without periodic normal events."""

    def __init__(self, camera: CameraModule, store: EventStore):
        self._camera = camera
        self._store  = store

        from ultralytics import YOLO
        model_path = Detector._resolve_model(YOLO_MODEL)
        logger.info('Cargando modelo YOLO...')
        self._model = YOLO(model_path)
        logger.info('Modelo YOLO listo')

        self._no_helmet_since: Optional[float] = None
        self._alert_triggered                  = False
        self._clear_counter                    = 0

    def run(self) -> None:
        try:
            while True:
                now   = time.time()
                frame = self._camera.latest_frame()

                if frame is not None:
                    detection = self._infer(frame)
                    no_helmet = detection is not None

                    if no_helmet:
                        self._clear_counter = 0
                        if self._no_helmet_since is None:
                            self._no_helmet_since = now
                            logger_cam.warning('Sin casco detectado — iniciando conteo')
                        elapsed = now - self._no_helmet_since
                        if elapsed >= ALERT_DURATION_SECS and not self._alert_triggered:
                            logger_cam.warning(f'Sin casco por {elapsed:.0f}s — disparando alerta')
                            self._trigger_alert(detection)
                            self._alert_triggered = True
                    else:
                        self._clear_counter += 1
                        if self._clear_counter >= 3:
                            if self._no_helmet_since is not None:
                                logger_cam.info('Casco detectado — reseteando conteo')
                            self._no_helmet_since = None
                            self._alert_triggered = False

                time.sleep(DETECTION_INTERVAL)
        except KeyboardInterrupt:
            pass

    def _infer(self, frame: np.ndarray) -> Optional[tuple[int, int]]:
        _NO_HELMET_LABEL = 'NO-Hardhat'
        results = self._model(frame, verbose=False)
        best_conf = -1.0
        best_xy: Optional[tuple[int, int]] = None
        for r in results:
            for box in r.boxes:
                label = r.names[int(box.cls)]
                conf  = float(box.conf)
                if label == _NO_HELMET_LABEL and conf >= YOLO_CONF and conf > best_conf:
                    best_conf = conf
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    best_xy = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        return best_xy

    def _trigger_alert(self, detection: tuple[int, int]) -> None:
        event_id  = str(uuid.uuid4())
        partition = datetime.now(timezone.utc).strftime('%Y/%m/%d/%H')
        temperature = read_temperature()
        photo_path = self._camera.sacar_foto(PENDING_DIR)
        video_path = self._camera.grabar_clip(PENDING_DIR, seconds=ALERT_DURATION_SECS)
        payload = {
            'event_id':    event_id,
            'type':        'alert',
            'device_id':   DEVICE_ID,
            'is_alert':    True,
            'no_helmet':   None,
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
        logger_cam.warning(f'Alerta encolada: event_id={event_id}')


def main() -> None:
    logger.info(f'Iniciando alerts-only device={DEVICE_ID} camera={CAMERA_DEVICE}')

    camera = CameraModule(CAMERA_DEVICE, CAMERA_BUFFER_SECS)
    camera.start()
    start_camera_server(camera)

    store    = EventStore(DATA_DIR / 'events.db', PENDING_DIR)
    detector = AlertOnlyDetector(camera, store)

    threading.Thread(target=queue_worker, args=(store,), daemon=True).start()
    detector.run()


if __name__ == '__main__':
    main()
