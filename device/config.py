import os
from pathlib import Path
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
KIT_DIR   = REPO_ROOT / 'connection_kit'

load_dotenv(REPO_ROOT / '.env')

# Paths
DATA_DIR    = REPO_ROOT / 'data'
PENDING_DIR = DATA_DIR / 'pending'

# MQTT – X.509 (connection_kit/)
MQTT_HOST  = os.getenv('AWS_IOT_ENDPOINT')
MQTT_PORT  = int(os.getenv('MQTT_PORT', 8883))
DEVICE_ID      = os.getenv('DEVICE_ID', 'raspi')
MQTT_TOPIC     = os.getenv('MQTT_TOPIC', f'g2/{DEVICE_ID}/events')
MQTT_CMD_TOPIC = os.getenv('MQTT_CMD_TOPIC', f'g2/{DEVICE_ID}/commands')
MQTT_CERT  = str(KIT_DIR / 'raspi.cert.pem')
MQTT_KEY   = str(KIT_DIR / 'raspi.private.key')
MQTT_CA    = str(KIT_DIR / 'root-CA.crt')

# Cognito M2M – para presigned URL (HTTP API)
TOKEN_URL     = os.getenv('COGNITO_TOKEN_URL')
CLIENT_ID     = os.getenv('COGNITO_CLIENT_ID')
CLIENT_SECRET = os.getenv('COGNITO_CLIENT_SECRET')
COGNITO_SCOPE = os.getenv('COGNITO_SCOPE')

# API Gateway
API_URL       = os.getenv('API_URL')
WEBSOCKET_URL = os.getenv('WEBSOCKET_URL')

# AWS Resources
FIREHOSE_STREAM   = os.getenv('FIREHOSE_STREAM')
S3_DATA_BUCKET    = os.getenv('S3_DATA_BUCKET')
S3_MEDIA_BUCKET   = os.getenv('S3_MEDIA_BUCKET')
SNS_ALERTS_TOPIC  = os.getenv('SNS_ALERTS_TOPIC')
COGNITO_USER_POOL = os.getenv('COGNITO_USER_POOL')

# Cámara
CAMERA_DEVICE      = int(os.getenv('CAMERA_DEVICE', 0))
CAMERA_BUFFER_SECS = int(os.getenv('CAMERA_BUFFER_SECS', 20))
CAMERA_PORT        = int(os.getenv('CAMERA_PORT', 5000))
CAMERA_WIDTH       = int(os.getenv('CAMERA_WIDTH', 1280))
CAMERA_HEIGHT      = int(os.getenv('CAMERA_HEIGHT', 720))

# Detector
YOLO_MODEL             = os.getenv('YOLO_MODEL', 'models/helmet_model_medium.pt')
YOLO_CONF              = float(os.getenv('YOLO_CONF', 0.40))
DETECTION_INTERVAL     = float(os.getenv('DETECTION_INTERVAL', 1.0))
ALERT_DURATION_SECS    = int(os.getenv('ALERT_DURATION_SECS', 3))
ALERT_REPEAT_INTERVAL  = float(os.getenv('ALERT_REPEAT_INTERVAL', 15.0))
NORMAL_EVENT_INTERVAL  = float(os.getenv('NORMAL_EVENT_INTERVAL', 300.0))

PERSON_MODEL_PATH      = os.getenv('PERSON_MODEL_PATH', 'yolov8n.pt')
IOU_THRESHOLD          = float(os.getenv('IOU_THRESHOLD', 0.45))
IMG_SIZE               = int(os.getenv('IMG_SIZE', 640))
PROCESS_EVERY_N_FRAMES = int(os.getenv('PROCESS_EVERY_N_FRAMES', 3))
USE_PERSON_FALLBACK    = os.getenv('USE_PERSON_FALLBACK', 'true').lower() == 'true'
USE_COLOR_FALLBACK     = os.getenv('USE_COLOR_FALLBACK', 'true').lower() == 'true'
SAVE_OUTPUT            = os.getenv('SAVE_OUTPUT', 'true').lower() == 'true'
OUTPUT_PATH            = os.getenv('OUTPUT_PATH', 'output_detection.mp4')
DISPLAY                = os.getenv('DISPLAY_VIDEO', 'true').lower() == 'true'
