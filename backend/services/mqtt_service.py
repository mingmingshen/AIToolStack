"""MQTT service: subscribe to images uploaded by devices"""
import json
import base64
import uuid
from pathlib import Path
from datetime import datetime
from typing import Optional
import paho.mqtt.client as mqtt
from PIL import Image as PILImage
import io

from backend.config import settings
from backend.models.database import SessionLocal, Image, Project
from backend.services.websocket_manager import websocket_manager
from backend.services.mqtt_broker import builtin_mqtt_broker


class MQTTService:
    """MQTT subscription service"""
    
    def __init__(self):
        self.client: Optional[mqtt.Client] = None
        self.is_connected = False
        self.broker_host = ""  # Save current connected broker address
        self.broker_port = 0
    
    def on_connect(self, client, userdata, flags, rc):
        """Connection callback"""
        if rc == 0:
            self.is_connected = True
            print(f"[MQTT] Connected to broker at {self.broker_host}:{self.broker_port}")
            # Subscribe to upload topic
            client.subscribe(settings.MQTT_UPLOAD_TOPIC, qos=settings.MQTT_QOS)
            print(f"[MQTT] Subscribed to topic: {settings.MQTT_UPLOAD_TOPIC}")
        else:
            print(f"[MQTT] Connection failed with code {rc}")
    
    def on_disconnect(self, client, userdata, rc):
        """Disconnect callback"""
        self.is_connected = False
        if rc != 0:
            print(f"[MQTT] Disconnected from broker unexpectedly (rc={rc})")
        else:
            print(f"[MQTT] Disconnected from broker")
        
        # If abnormal disconnect (rc != 0), paho-mqtt will automatically try to reconnect
        # We don't need to manually handle reconnection logic
    
    def on_message(self, client, userdata, msg):
        """Message receive callback"""
        try:
            topic = msg.topic
            payload = msg.payload.decode('utf-8')
            
            # Parse topic to get project_id
            # Topic format: annotator/upload/{project_id}
            parts = topic.split('/')
            if len(parts) < 3:
                print(f"[MQTT] Invalid topic format: {topic}")
                return
            
            project_id = parts[2]
            
            # Parse JSON payload
            data = json.loads(payload)
            
            # Handle image upload
            self._handle_image_upload(project_id, data, topic)
            
        except json.JSONDecodeError as e:
            print(f"[MQTT] JSON decode error: {e}")
            self._send_error_response(data.get('req_id', ''), data.get('device_id', ''), 
                                     "Invalid JSON format")
        except Exception as e:
            print(f"[MQTT] Error processing message: {e}")
            self._send_error_response(data.get('req_id', ''), data.get('device_id', ''), 
                                     str(e))
    
    def _handle_image_upload(self, project_id: str, data: dict, topic: str):
        """Handle image upload"""
        # Adapt to new data structure
        # Support two formats:
        # 1. New format: { "image_data": "...", "encoding": "...", "metadata": {...} }
        # 2. Old format: { "req_id": "...", "device_id": "...", "image": {...} }
        
        # Try new format
        if 'image_data' in data:
            # New format
            req_id = data.get('req_id', str(uuid.uuid4()))
            device_id = data.get('device_id', topic.split('/')[-1] if '/' in topic else 'unknown')
            metadata = data.get('metadata', {})
            encoding = data.get('encoding', 'base64')
            base64_data = data.get('image_data', '')
            
            # Extract information from metadata
            image_id = metadata.get('image_id', f'img_{int(datetime.utcnow().timestamp())}')
            timestamp = metadata.get('timestamp', int(datetime.utcnow().timestamp()))
            image_format = metadata.get('format', 'jpeg').lower()
            # If metadata has dimension information, use it first
            metadata_width = metadata.get('width')
            metadata_height = metadata.get('height')
            
            # Generate filename
            if image_format in ['jpeg', 'jpg']:
                filename = f'{image_id}.jpg'
            elif image_format == 'png':
                filename = f'{image_id}.png'
            else:
                filename = f'{image_id}.{image_format}'
        else:
            # Old format (backward compatible)
            req_id = data.get('req_id', str(uuid.uuid4()))
            device_id = data.get('device_id', 'unknown')
            timestamp = data.get('timestamp', int(datetime.utcnow().timestamp()))
            image_data = data.get('image', {})
            metadata = data.get('metadata', {})
            
            filename = image_data.get('filename', f'img_{timestamp}.jpg')
            image_format = image_data.get('format', 'jpg').lower()
            encoding = image_data.get('encoding', 'base64')
            base64_data = image_data.get('data', '')
            metadata_width = None
            metadata_height = None
        
        # Verify project exists
        db = SessionLocal()
        try:
            project = db.query(Project).filter(Project.id == project_id).first()
            if not project:
                error_msg = f"Project {project_id} not found"
                print(f"[MQTT] {error_msg}")
                self._send_error_response(req_id, device_id, error_msg)
                return
            
            # Process base64 data, remove possible data URI prefix
            # Supported formats:
            # 1. data:image/jpeg;base64,xxxxx
            # 2. data:image/png;base64,xxxxx
            # 3. data:image/jpg;base64,xxxxx
            # 4. Pure base64 string
            if base64_data.startswith('data:'):
                # Contains data URI prefix, extract base64 part
                if ',' in base64_data:
                    base64_data = base64_data.split(',')[-1]
                else:
                    # If format is abnormal, try to remove data: prefix
                    base64_data = base64_data.replace('data:', '').split(';')[-1]
            elif ',' in base64_data:
                # Might contain other separators
                base64_data = base64_data.split(',')[-1]
            
            # Clean possible whitespace characters
            base64_data = base64_data.strip()
            
            # Base64 decode
            if encoding != 'base64':
                raise ValueError(f"Unsupported encoding: {encoding}")
            
            try:
                image_bytes = base64.b64decode(base64_data)
            except Exception as e:
                raise ValueError(f"Failed to decode base64 data: {str(e)}")
            
            # Verify image size
            size_mb = len(image_bytes) / (1024 * 1024)
            if size_mb > settings.MAX_IMAGE_SIZE_MB:
                raise ValueError(f"Image too large: {size_mb:.2f}MB (max: {settings.MAX_IMAGE_SIZE_MB}MB)")
            
            # Get image dimensions
            if metadata_width and metadata_height:
                # Use dimension information from metadata
                img_width = metadata_width
                img_height = metadata_height
            else:
                # Open image to get dimensions
                img = PILImage.open(io.BytesIO(image_bytes))
                img_width, img_height = img.size
            
            # Generate storage path
            project_dir = settings.DATASETS_ROOT / project_id / "raw"
            project_dir.mkdir(parents=True, exist_ok=True)
            
            # Handle filename conflicts
            file_path = project_dir / filename
            if file_path.exists():
                stem = file_path.stem
                suffix = file_path.suffix
                timestamp_suffix = int(datetime.utcnow().timestamp())
                filename = f"{stem}_{timestamp_suffix}{suffix}"
                file_path = project_dir / filename
            
            # Save image
            file_path.write_bytes(image_bytes)
            
            # Generate relative path (only includes raw/filename, not project_id)
            relative_path = f"raw/{filename}"
            
            # Save to database
            db_image = Image(
                project_id=project_id,
                filename=filename,
                path=relative_path,
                width=img_width,
                height=img_height,
                status="UNLABELED",
                source=f"MQTT:{device_id}"
            )
            db.add(db_image)
            db.commit()
            db.refresh(db_image)
            
            print(f"[MQTT] Image saved: {filename} ({img_width}x{img_height}) to project {project_id}")
            
            # Send success response
            self._send_success_response(req_id, device_id, project_id)
            
            # Notify frontend via WebSocket
            websocket_manager.broadcast_project_update(project_id, {
                "type": "new_image",
                "image_id": db_image.id,
                "filename": filename,
                "path": relative_path,
                "width": img_width,
                "height": img_height
            })
            
        except Exception as e:
            db.rollback()
            error_msg = f"Failed to save image: {str(e)}"
            print(f"[MQTT] {error_msg}")
            self._send_error_response(req_id, device_id, error_msg)
        finally:
            db.close()
    
    def _send_success_response(self, req_id: str, device_id: str, project_id: str):
        """Send success response"""
        if not device_id or device_id == 'unknown':
            return
        
        response_topic = f"{settings.MQTT_RESPONSE_TOPIC_PREFIX}/{device_id}"
        response = {
            "req_id": req_id,
            "status": "success",
            "code": 200,
            "message": f"Image saved to project {project_id}",
            "server_time": int(datetime.utcnow().timestamp())
        }
        
        self.client.publish(response_topic, json.dumps(response), qos=settings.MQTT_QOS)
    
    def _send_error_response(self, req_id: str, device_id: str, error_message: str):
        """Send error response"""
        if not device_id or device_id == 'unknown':
            return
        
        response_topic = f"{settings.MQTT_RESPONSE_TOPIC_PREFIX}/{device_id}"
        response = {
            "req_id": req_id,
            "status": "error",
            "code": 400,
            "message": error_message,
            "server_time": int(datetime.utcnow().timestamp())
        }
        
        self.client.publish(response_topic, json.dumps(response), qos=settings.MQTT_QOS)
    
    def start(self):
        """Start MQTT client"""
        if not settings.MQTT_ENABLED:
            print("[MQTT] MQTT service is disabled in configuration")
            return
        
        try:
            # Explicitly specify MQTT 3.1.1 protocol (aMQTT broker doesn't support MQTT 5.0)
            # protocol=mqtt.MQTTv311 means using MQTT 3.1.1
            # Set clean_session=True to ensure connection is clean
            self.client = mqtt.Client(
                client_id=f"annotator_server_{uuid.uuid4().hex[:8]}",
                protocol=mqtt.MQTTv311,
                clean_session=True
            )
            
            # Set connection timeout and retry parameters
            self.client.reconnect_delay_set(min_delay=1, max_delay=120)
            
            # Set callbacks
            self.client.on_connect = self.on_connect
            self.client.on_disconnect = self.on_disconnect
            self.client.on_message = self.on_message
            
            # Determine Broker address to connect to
            if settings.MQTT_USE_BUILTIN_BROKER:
                # Built-in Broker binds to 0.0.0.0, client in same container should connect to localhost
                # This is more reliable, avoids connection issues caused by container internal IP
                self.broker_host = "127.0.0.1"  # Use localhost connection in container
                self.broker_port = settings.MQTT_BUILTIN_PORT
                print(f"[MQTT] Using built-in MQTT Broker at {self.broker_host}:{self.broker_port}")
            else:
                self.broker_host = settings.MQTT_BROKER
                self.broker_port = settings.MQTT_PORT
                print(f"[MQTT] Connecting to external MQTT Broker at {self.broker_host}:{self.broker_port}")
                # External Broker requires authentication
                if settings.MQTT_USERNAME and settings.MQTT_PASSWORD:
                    self.client.username_pw_set(settings.MQTT_USERNAME, settings.MQTT_PASSWORD)
            
            # Connect to Broker
            # Increase keepalive time to reduce timeout disconnection issues
            self.client.connect(self.broker_host, self.broker_port, keepalive=120)
            self.client.loop_start()
        except ConnectionRefusedError:
            if settings.MQTT_USE_BUILTIN_BROKER:
                print(f"[MQTT] Connection refused. Built-in broker may not be running.")
            else:
                print(f"[MQTT] Connection refused. Please check if MQTT broker is running at {settings.MQTT_BROKER}:{settings.MQTT_PORT}")
            self.is_connected = False
        except Exception as e:
            print(f"[MQTT] Failed to connect: {e}")
            self.is_connected = False
    
    def stop(self):
        """Stop MQTT client"""
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()
            self.is_connected = False


# Global MQTT service instance
mqtt_service = MQTTService()

