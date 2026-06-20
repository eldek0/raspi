import json
import logging
import os
import ssl
import threading
import time
import uuid
from datetime import datetime, timezone

import paho.mqtt.client as mqtt
import requests

logger = logging.getLogger(__name__)

from auth_manager import CognitoAuth
from config import (
    MQTT_HOST, MQTT_PORT, MQTT_TOPIC, MQTT_CMD_TOPIC, DEVICE_ID,
    MQTT_CERT, MQTT_KEY, MQTT_CA,
    TOKEN_URL, CLIENT_ID, CLIENT_SECRET, COGNITO_SCOPE,
    API_URL, PENDING_DIR,
)
from store import EventStore

auth = CognitoAuth(
    token_url=TOKEN_URL,
    client_id=CLIENT_ID,
    client_secret=CLIENT_SECRET,
    scope=COGNITO_SCOPE,
)

_mqtt_client = None
_mqtt_lock   = threading.Lock()
_msg_handler = None  # callback set by start_command_listener


def _ensure_root_ca() -> None:
    if not os.path.exists(MQTT_CA):
        resp = requests.get('https://www.amazontrust.com/repository/AmazonRootCA1.pem', timeout=10)
        resp.raise_for_status()
        with open(MQTT_CA, 'w') as f:
            f.write(resp.text)


def _on_connect(client, userdata, flags, *args):
    rc = args[0] if args else '?'
    logger.info(f'MQTT conectado (rc={rc}) — suscribiendo a {MQTT_CMD_TOPIC}')
    client.subscribe(MQTT_CMD_TOPIC, qos=1)


def _on_disconnect(client, userdata, *args):
    rc = args[0] if args else '?'
    logger.warning(f'MQTT desconectado (rc={rc}) — reconectando automáticamente')


def _on_message(client, userdata, msg):
    global _msg_handler
    if _msg_handler is not None:
        _msg_handler(msg)


def _build_mqtt_client():
    try:
        client = mqtt.Client(
            client_id=DEVICE_ID,
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        )
    except Exception:
        client = mqtt.Client(client_id=DEVICE_ID)

    client.on_connect    = _on_connect
    client.on_disconnect = _on_disconnect
    client.on_message    = _on_message
    client.tls_set(
        ca_certs=MQTT_CA,
        certfile=MQTT_CERT,
        keyfile=MQTT_KEY,
        tls_version=ssl.PROTOCOL_TLSv1_2,
    )
    client.tls_insecure_set(False)
    client.reconnect_delay_set(min_delay=1, max_delay=30)
    return client


def _get_mqtt():
    global _mqtt_client
    with _mqtt_lock:
        if _mqtt_client is None:
            _ensure_root_ca()
            client = _build_mqtt_client()
            logger.info(f'Conectando MQTT a {MQTT_HOST}:{MQTT_PORT} como {DEVICE_ID}')
            client.connect_async(MQTT_HOST, MQTT_PORT, keepalive=60)
            client.loop_start()
            _mqtt_client = client
        return _mqtt_client


def send_mqtt_event(payload: dict) -> None:
    client = _get_mqtt()
    # wait up to 5s for reconnect before failing
    for _ in range(10):
        if client.is_connected():
            break
        time.sleep(0.5)
    if not client.is_connected():
        raise ConnectionError('MQTT desconectado')
    info = client.publish(MQTT_TOPIC, json.dumps(payload), qos=1)
    info.wait_for_publish(timeout=10)


def upload_photo(filepath: str, filename: str) -> str:
    size = os.path.getsize(filepath)
    logger.info(f'Subiendo foto: {filename} ({size} bytes)')
    resp = requests.get(
        f'{API_URL}/presigned-url',
        params={'filename': filename, 'device_id': DEVICE_ID},
        headers=auth.auth_header(),
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    with open(filepath, 'rb') as f:
        requests.put(data['url'], data=f, headers={'Content-Type': 'image/jpeg'}, timeout=30).raise_for_status()
    logger.info(f'Foto subida: key={data["key"]}')
    return data['key']


def upload_video(filepath: str, filename: str) -> str:
    size = os.path.getsize(filepath)
    logger.info(f'Subiendo video: {filename} ({size} bytes)')
    resp = requests.get(
        f'{API_URL}/presigned-url',
        params={'filename': filename, 'device_id': DEVICE_ID},
        headers=auth.auth_header(),
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json()
    with open(filepath, 'rb') as f:
        requests.put(data['url'], data=f, headers={'Content-Type': 'video/mp4'}, timeout=60).raise_for_status()
    logger.info(f'Video subido: key={data["key"]}')
    return data['key']



def start_command_listener(camera, store: EventStore, get_detection=None) -> None:
    """Registra handler de comandos sobre el cliente MQTT compartido."""
    global _msg_handler

    def handle(msg):
        try:
            payload = json.loads(msg.payload)
        except Exception:
            logging.warning(f'Comando no parseable: {msg.payload}')
            return

        action = payload.get('action')
        logger.info(f'Comando recibido: action={action} topic={msg.topic}')

        if action != 'screenshot':
            logger.debug(f'Comando ignorado: action={action}')
            return

        try:
            event_id   = str(uuid.uuid4())
            partition  = datetime.now(timezone.utc).strftime('%Y/%m/%d/%H')
            filepath   = camera.sacar_foto(PENDING_DIR)
            filename   = f'{partition}/{event_id}.jpg'
            detection  = get_detection() if get_detection else {}
            store.enqueue(
                'photo',
                event_id=event_id,
                data={
                    'event_id':    event_id,
                    'type':        'screenshot',
                    'device_id':   DEVICE_ID,
                    'is_alert':    detection.get('is_alert', False),
                    'no_helmet':   detection.get('is_alert', False),
                    'detection_x': detection.get('detection_x'),
                    'detection_y': detection.get('detection_y'),
                    'timestamp':   datetime.now(timezone.utc).isoformat(),
                    'action':      'screenshot_done',
                },
                filepath=filepath,
                filename=filename,
                priority=1,
            )
            logger.info(f'Screenshot encolado: {filename}')
        except Exception as e:
            logger.error(f'Error tomando screenshot: {e}')

    _msg_handler = handle
    _get_mqtt()  # fuerza conexión y suscripción via _on_connect
    logger.info('Command listener iniciado')


def queue_worker(store: EventStore) -> None:
    pending = store.count_pending()
    if pending:
        logger.info(f'Queue worker iniciado — {pending} items pendientes')
    else:
        logger.info('Queue worker iniciado — cola vacía')

    delay = 0
    while True:
        row = store.next_pending()
        if row is None:
            time.sleep(2)
            continue
        try:
            if row['type'] == 'event':
                if store.has_pending_sibling(row['event_id']):
                    store.increment_attempts(row['id'])
                    time.sleep(2)
                    continue
                logger.debug(f'Enviando evento MQTT: event_id={row["event_id"]}')
                send_mqtt_event(json.loads(row['data']))
                logger.info(f'Evento MQTT enviado: event_id={row["event_id"]}')
            elif row['type'] == 'photo':
                image_key = upload_photo(row['filepath'], row['filename'])
                os.unlink(row['filepath'])
                if row['data']:
                    payload = json.loads(row['data'])
                    payload['image_key'] = image_key
                    store.enqueue('event', event_id=row['event_id'], data=payload)
            elif row['type'] == 'video':
                video_key = upload_video(row['filepath'], row['filename'])
                os.unlink(row['filepath'])
                store.add_to_data(row['event_id'], 'event', {'video_key': video_key})
            store.mark_done(row['id'])
            delay = 0
        except FileNotFoundError as e:
            # File was deleted externally — retrying will never help, skip it
            logger.warning(
                f'Archivo no encontrado, descartando item de cola '
                f'({row["type"]}) id={row["id"]}: {e}'
            )
            store.mark_done(row['id'])
            delay = 0
        except Exception as e:
            store.increment_attempts(row['id'])
            delay = min(delay * 2 or 5, 300)
            logger.error(
                f'Error en cola ({row["type"]}) intento #{row["attempts"] + 1}: {e} '
                f'— reintentando en {delay}s'
            )
            time.sleep(delay)
