#!/usr/bin/env python3
"""Graba la pantalla en Wayland con wf-recorder."""
import os
import signal
import subprocess
import sys
from datetime import datetime


def main():
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output    = f'screen_{timestamp}.mp4'

    cmd = ['wf-recorder', '-f', output]

    print(f'Guardando en: {output}')
    print('Presioná Ctrl+C para detener.\n')

    proc = subprocess.Popen(cmd, start_new_session=True)

    def _stop(sig, frame):
        print('\nDeteniendo grabación...')
        proc.terminate()
        proc.wait()
        print(f'Video guardado: {output}')
        sys.exit(0)

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    proc.wait()


if __name__ == '__main__':
    main()