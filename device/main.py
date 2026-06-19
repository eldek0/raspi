import logging
import threading

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s %(message)s')
logging.getLogger().setLevel(logging.INFO)

logger = logging.getLogger(__name__)

from camera import CameraModule
from config import (
    CAMERA_DEVICE, CAMERA_BUFFER_SECS, DATA_DIR, PENDING_DIR, DEVICE_ID,
    YOLO_MODEL, YOLO_CONF, IOU_THRESHOLD, IMG_SIZE, PROCESS_EVERY_N_FRAMES,
    USE_PERSON_FALLBACK, USE_COLOR_FALLBACK, PERSON_MODEL_PATH,
    SAVE_OUTPUT, OUTPUT_PATH, DISPLAY,
)
from detector import Detector
from store import EventStore
from worker import queue_worker, start_command_listener


def main() -> None:
    logger.info(f'Iniciando device={DEVICE_ID} camera={CAMERA_DEVICE}')
    logger.info(
        f'Detector config: model={YOLO_MODEL} conf={YOLO_CONF} iou={IOU_THRESHOLD} '
        f'img_size={IMG_SIZE} every_n={PROCESS_EVERY_N_FRAMES} '
        f'person_fallback={USE_PERSON_FALLBACK} color_fallback={USE_COLOR_FALLBACK} '
        f'person_model={PERSON_MODEL_PATH}'
    )
    logger.info(
        f'Output config: save={SAVE_OUTPUT} path={OUTPUT_PATH} display={DISPLAY}'
    )

    camera = CameraModule(CAMERA_DEVICE, CAMERA_BUFFER_SECS)
    camera.start()

    store    = EventStore(DATA_DIR / 'events.db', PENDING_DIR)
    detector = Detector(camera, store)

    threading.Thread(target=queue_worker, args=(store,), daemon=True).start()
    start_command_listener(camera, store)
    detector.run()


if __name__ == '__main__':
    main()
