"""Screenshot-command listener only — no model, no sensor, no periodic events.
Blocks waiting for MQTT action=screenshot commands."""
import logging
import signal
import threading

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')

logger = logging.getLogger(__name__)

from camera import CameraModule
from config import (
    CAMERA_DEVICE, CAMERA_BUFFER_SECS, DATA_DIR, PENDING_DIR, DEVICE_ID,
)
from store import EventStore
from worker import queue_worker, start_command_listener


def main() -> None:
    logger.info(f'Iniciando screenshot-listener device={DEVICE_ID}')

    camera = CameraModule(CAMERA_DEVICE, CAMERA_BUFFER_SECS)
    camera.start()

    store = EventStore(DATA_DIR / 'events.db', PENDING_DIR)
    threading.Thread(target=queue_worker, args=(store,), daemon=True).start()

    if not camera.wait_ready(timeout=15.0):
        logger.warning('Cámara no lista en 15s — continuando de todas formas')

    start_command_listener(camera, store)
    logger.info('Esperando comandos screenshot...')

    stop = threading.Event()
    signal.signal(signal.SIGINT,  lambda *_: stop.set())
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    stop.wait()


if __name__ == '__main__':
    main()
