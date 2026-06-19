import argparse
import logging
import threading
import time
import uuid
from datetime import datetime, timezone

from PIL import Image, ImageDraw

from config import DEVICE_ID, DATA_DIR, PENDING_DIR
from store import EventStore
from worker import queue_worker

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

store = EventStore(DATA_DIR / 'events.db', PENDING_DIR)


# ── Archivos sintéticos ───────────────────────────────────────────────────────

def _make_photo(ts: str, epoch: int) -> str:
    img  = Image.new('RGB', (640, 480), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, 640, 60], fill=(200, 0, 0))
    draw.text((20, 15), 'ALERTA: SIN CASCO', fill='white')
    draw.text((20, 100), f'Dispositivo: {DEVICE_ID}', fill=(200, 200, 200))
    draw.text((20, 140), ts, fill=(200, 200, 200))
    path = str(PENDING_DIR / f'photo_{epoch}.jpg')
    img.save(path, 'JPEG')
    return path


def _make_video(epoch: int) -> str:
    path = str(PENDING_DIR / f'video_{epoch}.mp4')
    with open(path, 'wb') as f:
        f.write(b'\x00' * 1024 * 10)
    return path


# ── Generadores de eventos ────────────────────────────────────────────────────

def generate_alert() -> None:
    event_id  = str(uuid.uuid4())
    ts        = datetime.now(timezone.utc).isoformat()
    epoch     = int(time.time())
    partition = datetime.now(timezone.utc).strftime('%Y/%m/%d/%H')

    payload = {
        'event_id':  event_id,
        'type':      'alert',
        'device_id': DEVICE_ID,
        'is_alert':  True,
        'no_helmet': True,
        'timestamp': ts,
    }
    photo_path = _make_photo(ts, epoch)
    video_path = _make_video(epoch)

    store.enqueue('photo', event_id=event_id, data=payload, filepath=photo_path, filename=f'{partition}/{event_id}.jpg')
    store.enqueue('video', event_id=event_id, filepath=video_path, filename=f'{partition}/{event_id}.mp4')
    logging.warning(f'[ALERTA] event_id={event_id} — foto + video encolados')


def generate_normal_event() -> None:
    event_id = str(uuid.uuid4())
    payload  = {
        'event_id':  event_id,
        'type':      'normal',
        'device_id': DEVICE_ID,
        'is_alert':  False,
        'no_helmet': False,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }
    store.enqueue('event', event_id=event_id, data=payload)
    logging.info(f'[NORMAL] event_id={event_id} encolado')


# ── Loop principal ────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description='Simulador de dispositivo Raspi')
    parser.add_argument(
        '--mode',
        choices=['full', 'events', 'alerts'],
        default='full',
        help='full: eventos + alertas (default) | events: solo eventos normales | alerts: solo alertas',
    )
    args = parser.parse_args()

    threading.Thread(target=queue_worker, args=(store,), daemon=True).start()
    logging.info(f'Simulador iniciado en modo [{args.mode}]. Ctrl+C para detener.')

    last_normal: float = 0.0
    last_alert:  float = 0.0

    ALERT_INTERVAL  = 10.0
    NORMAL_INTERVAL = 10.0
    TICK            = 2.0

    try:
        while True:
            now = time.time()

            if args.mode in ('full', 'events') and now - last_normal >= NORMAL_INTERVAL:
                generate_normal_event()
                last_normal = now

            if args.mode in ('full', 'alerts') and now - last_alert >= ALERT_INTERVAL:
                generate_alert()
                last_alert = now

            time.sleep(TICK)
    except KeyboardInterrupt:
        logging.info('Simulador detenido.')


if __name__ == '__main__':
    main()
