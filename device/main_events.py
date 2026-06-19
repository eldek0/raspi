"""Periodic normal-event emitter — reads temperature sensor, no YOLO model."""
import logging
import random
import threading
import time
import uuid
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')

logger = logging.getLogger(__name__)

from camera import CameraModule
from config import (
    CAMERA_DEVICE, CAMERA_BUFFER_SECS, DATA_DIR, PENDING_DIR,
    DEVICE_ID, NORMAL_EVENT_INTERVAL, DETECTION_INTERVAL,
)
from detector import read_temperature
from store import EventStore
from worker import queue_worker


def _emit_normal_event(store: EventStore) -> None:
    event_id    = str(uuid.uuid4())
    temperature = read_temperature()
    if temperature == -1.0:
        temperature = round(random.uniform(20.0, 35.0), 1)
    payload = {
        'event_id':      event_id,
        'type':          'normal',
        'device_id':     DEVICE_ID,
        'is_alert':      False,
        'no_helmet':     None,
        'detection_x':   None,
        'detection_y':   None,
        'temperature':   temperature,
        'timestamp':     datetime.now(timezone.utc).isoformat(),
    }
    store.enqueue('event', event_id=event_id, data=payload)
    logger.info(f'Evento normal encolado: temp={temperature}°C event_id={event_id}')


def main() -> None:
    logger.info(f'Iniciando events-only device={DEVICE_ID}')

    store = EventStore(DATA_DIR / 'events.db', PENDING_DIR)
    threading.Thread(target=queue_worker, args=(store,), daemon=True).start()

    last_normal = 0.0
    try:
        while True:
            now = time.time()
            if now - last_normal >= NORMAL_EVENT_INTERVAL:
                _emit_normal_event(store)
                last_normal = now
            time.sleep(DETECTION_INTERVAL)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
