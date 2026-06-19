import os
import re
import threading
import time
import logging
from collections import deque
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from config import CAMERA_DEVICE, CAMERA_BUFFER_SECS, CAMERA_PORT, CAMERA_WIDTH, CAMERA_HEIGHT

logger = logging.getLogger(__name__)


class CameraModule:
    def __init__(self, device: int = CAMERA_DEVICE, buffer_seconds: int = CAMERA_BUFFER_SECS):
        self._device         = device
        self._buffer_seconds = buffer_seconds
        self._buffer: deque  = deque()
        self._lock           = threading.Lock()
        self._stop_event     = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._error: Optional[str] = None
        self._thread  = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def status(self) -> dict:
        # len(deque) es atómico en CPython — no necesita lock
        frames = len(self._buffer)
        alive  = self._thread is not None and self._thread.is_alive()
        return {
            'thread_alive': alive,
            'frames_buffered': frames,
            'device': self._device,
            'error': getattr(self, '_error', None),
        }

    def wait_ready(self, timeout: float = 10.0) -> bool:
        """Block until at least one frame is buffered or timeout expires. Returns True if ready."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._buffer:
                    return True
            if getattr(self, '_error', None):
                return False
            time.sleep(0.1)
        return False

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _capture_loop(self) -> None:
        cap = cv2.VideoCapture(self._device)
        if not cap.isOpened():
            msg = f'No se pudo abrir la cámara device={self._device}. Probá otro índice con CAMERA_DEVICE=1 (o 2, etc.)'
            logger.error(msg)
            self._error = msg
            return

        logger.info(f'Cámara conectada: device={self._device}')
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        # Buffer mínimo para que cap.read() no bloquee más de un frame
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        try:
            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    if self._stop_event.wait(timeout=0.01):
                        break
                    continue

                ret2, jpg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                if not ret2:
                    continue

                ts = time.time()
                with self._lock:
                    self._buffer.append({'ts': ts, 'jpg': jpg.tobytes()})
                    cutoff = ts - (self._buffer_seconds * 1.5)
                    while self._buffer and self._buffer[0]['ts'] < cutoff:
                        self._buffer.popleft()
        finally:
            cap.release()
            logger.info(f'Cámara liberada: device={self._device}')

    def latest_frame(self) -> Optional[np.ndarray]:
        """Último frame como array numpy (para inferencia YOLO, sin I/O a disco)."""
        with self._lock:
            if not self._buffer:
                return None
            jpg = self._buffer[-1]['jpg']
        return cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

    def latest_frame_jpg(self) -> Optional[bytes]:
        """Último frame como JPEG bytes (para Flask /snapshot)."""
        with self._lock:
            if not self._buffer:
                return None
            return self._buffer[-1]['jpg']

    def _ensure_running(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            logger.warning('Capture thread muerto — reiniciando')
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._thread.start()

    def sacar_foto(self, dest_dir: Path) -> str:
        """Guarda el último frame como JPEG en dest_dir y devuelve el path."""
        self._ensure_running()
        if not self.wait_ready(timeout=10.0):
            raise RuntimeError('No hay frames disponibles en el buffer')
        jpg = self.latest_frame_jpg()
        if jpg is None:
            raise RuntimeError('No hay frames disponibles en el buffer')
        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        path = str(dest_dir / f'foto_{int(time.time())}.jpg')
        with open(path, 'wb') as f:
            f.write(jpg)
        logger.info(f'Foto guardada: {path} ({len(jpg)} bytes)')
        return path

    def grabar_clip(self, dest_dir: Path, seconds: int = 15) -> str:
        """Ensambla un MP4 con los últimos `seconds` segundos del buffer y devuelve el path."""
        now    = time.time()
        cutoff = now - seconds
        with self._lock:
            frames = [(f['ts'], f['jpg']) for f in self._buffer if f['ts'] >= cutoff]

        if not frames:
            raise RuntimeError('No hay frames suficientes en el buffer')

        # decodificar primer frame para obtener dimensiones
        arr0 = cv2.imdecode(np.frombuffer(frames[0][1], dtype=np.uint8), cv2.IMREAD_COLOR)
        h, w = arr0.shape[:2]

        # estimar FPS a partir de los timestamps reales
        times = [f[0] for f in frames]
        if len(times) >= 2:
            duration = (times[-1] - times[0]) or 0.033
            fps = max(1, int(round(len(times) / duration)))
        else:
            fps = 15

        dest_dir = Path(dest_dir)
        dest_dir.mkdir(parents=True, exist_ok=True)
        path = str(dest_dir / f'clip_{int(time.time())}.mp4')

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
        for _, jpg in frames:
            arr = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if arr is not None:
                writer.write(arr)
        writer.release()
        size = os.path.getsize(path)
        logger.info(f'Clip guardado: {path} ({len(frames)} frames, {size} bytes)')
        return path

    def _frames_since(self, seconds: int):
        now    = time.time()
        cutoff = now - seconds
        with self._lock:
            return [f['jpg'] for f in self._buffer if f['ts'] >= cutoff]

    def camera_info(self) -> dict:
        device   = self._device
        sys_path = f'/sys/class/video4linux/video{device}/name'
        info     = {'device': device, 'name': None, 'opened': False}
        try:
            if os.path.exists(sys_path):
                with open(sys_path, 'r', encoding='utf-8', errors='ignore') as f:
                    info['name']   = f.read().strip()
                    info['source'] = 'sysfs'
            else:
                cap    = cv2.VideoCapture(device)
                opened = cap.isOpened()
                info['opened'] = bool(opened)
                if opened:
                    get_backend = getattr(cap, 'getBackendName', None)
                    if callable(get_backend):
                        backend = get_backend()
                        if backend:
                            info['name']   = backend
                            info['source'] = 'backend'
                    if not info.get('name'):
                        ret, frame = cap.read()
                        if ret and frame is not None:
                            h, w = frame.shape[:2]
                            info['name']   = f'frame_{w}x{h}'
                            info['source'] = 'frame'
                cap.release()
        except Exception as e:
            info['error'] = str(e)
        return info


# ── Flask debug server (solo cuando se corre este archivo directamente) ───────

def _make_flask_app(camera: CameraModule):
    from flask import Flask, Response, jsonify, request, send_file
    import tempfile

    app = Flask(__name__)

    @app.route('/health')
    def health():
        return jsonify({'status': 'ok', 'camera': camera.status()})

    @app.route('/snapshot')
    def snapshot():
        jpg = camera.latest_frame_jpg()
        if jpg is None:
            return jsonify({'error': 'no frames yet'}), 503
        return Response(jpg, mimetype='image/jpeg')

    @app.route('/save_last', methods=['GET', 'POST'])
    def save_last():
        try:
            seconds = int(request.args.get('seconds', camera._buffer_seconds))
        except Exception:
            seconds = camera._buffer_seconds

        filename = None
        if request.method == 'GET':
            filename = request.args.get('filename')
        elif request.is_json:
            try:
                filename = request.json.get('filename')
            except Exception:
                pass
        else:
            filename = request.form.get('filename')

        try:
            tmp_dir = Path(tempfile.gettempdir())
            path    = camera.grabar_clip(tmp_dir, seconds=seconds)
        except RuntimeError as e:
            return jsonify({'error': str(e)}), 503

        def _secure_name(name):
            if not name:
                return None
            name = os.path.basename(name)
            if not name.lower().endswith('.mp4'):
                name += '.mp4'
            return re.sub(r'[^A-Za-z0-9._-]', '_', name)

        out_name = _secure_name(filename) or f'last_{seconds}s.mp4'
        return send_file(path, mimetype='video/mp4', as_attachment=True, download_name=out_name)

    @app.route('/camera_name')
    def camera_name():
        return jsonify(camera.camera_info())

    @app.route('/shutdown', methods=['POST'])
    def shutdown():
        camera.stop()
        return jsonify({'status': 'stopping'})

    return app


if __name__ == '__main__':
    import logging as _logging
    _logging.basicConfig(level=_logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

    _camera = CameraModule()
    _camera.start()

    _app = _make_flask_app(_camera)
    _app.run(host='0.0.0.0', port=CAMERA_PORT, threaded=True)
