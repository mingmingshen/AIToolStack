"""Configuration file"""
import socket
import os
from pathlib import Path
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).parent.parent
DATASETS_DIR = BASE_DIR / "datasets"


def get_local_ip() -> str:
    """Get local IP address (prefer returning non-container internal IP)"""
    
    # Fallback method 1: Get local IP by connecting to external address (standard method)
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.settimeout(0)
        try:
            s.connect(('8.8.8.8', 80))
            ip = s.getsockname()[0]
            s.close()
            # If obtained IP is not container internal IP or localhost, can use it
            container_ranges = ['172.17.', '172.18.', '172.19.', '172.20.', '172.21.', '172.22.', '172.23.', '172.24.', '172.25.', '172.26.', '172.27.', '172.28.', '172.29.', '172.30.', '172.31.']
            if ip != '127.0.0.1' and not any(ip.startswith(prefix) for prefix in container_ranges):
                return ip
        except Exception:
            pass
        finally:
            try:
                s.close()
            except:
                pass
    except Exception:
        pass
    
    # Fallback method 2: Use hostname
    try:
        ip = socket.gethostbyname(socket.gethostname())
        if ip != '127.0.0.1':
            container_ranges = ['172.17.', '172.18.', '172.19.']
            if not any(ip.startswith(prefix) for prefix in container_ranges):
                return ip
    except Exception:
        pass
    
    # If all fail, return localhost
    return "127.0.0.1"


def get_mqtt_broker_host(request=None) -> str:
    """
    Get MQTT Broker address displayed externally
    Prefer using configured MQTT_BROKER_HOST, if not available try to get from request headers
    """
    # If MQTT_BROKER_HOST is configured, use it directly
    if settings.MQTT_BROKER_HOST:
        return settings.MQTT_BROKER_HOST
    
    # If request object exists, prefer getting from request headers (reflects actual IP client accesses)
    if request:
        # Priority 1: Try to get from X-Forwarded-Host (reverse proxy scenario, most reliable)
        forwarded_host = request.headers.get("X-Forwarded-Host", "")
        if forwarded_host:
            host_without_port = forwarded_host.split(":")[0]
            if host_without_port not in ["localhost", "127.0.0.1", "0.0.0.0"]:
                return host_without_port
        
        # Priority 2: Try to get from Host header
        host = request.headers.get("Host", "")
        if host:
            # Remove port number (if exists)
            host_without_port = host.split(":")[0]
            # Exclude localhost and 127.0.0.1, but keep other addresses
            if host_without_port not in ["localhost", "127.0.0.1", "0.0.0.0"]:
                return host_without_port
        
        # Priority 3: Try to get from X-Real-IP (some reverse proxy settings)
        real_ip = request.headers.get("X-Real-IP", "")
        if real_ip:
            if real_ip not in ["localhost", "127.0.0.1", "0.0.0.0"]:
                return real_ip
    
    # Method 1: Try to get from environment variables (Docker Compose might set, or manual config)
    host_ip = os.environ.get("HOST_IP") or os.environ.get("HOSTIP") or os.environ.get("MQTT_BROKER_HOST") or os.environ.get("SERVER_IP")
    if host_ip:
        try:
            socket.inet_aton(host_ip)
            return host_ip
        except:
            pass
    
    # Method 2: Get local IP (improved, prefer returning non-container IP)
    local_ip = get_local_ip()
    
    # If it's container internal IP, try to get host IP
    container_ip_ranges = ["172.17.", "172.18.", "172.19.", "172.20.", "172.21.", "172.22.", "172.23.", "172.24.", "172.25.", "172.26.", "172.27.", "172.28.", "172.29.", "172.30.", "172.31."]
    is_container_ip = any(local_ip.startswith(prefix) for prefix in container_ip_ranges)
    
    if is_container_ip:
        # Method 2.1: In Docker container, try to get default gateway (usually Docker host)
        try:
            import subprocess
            result = subprocess.run(
                ["ip", "route", "show", "default"],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'default via' in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == 'via' and i + 1 < len(parts):
                                gateway_ip = parts[i + 1]
                                try:
                                    socket.inet_aton(gateway_ip)
                                    # Gateway IP is usually Docker host IP, but we need server external IP
                                    # Only use gateway IP when no other option
                                except:
                                    pass
        except Exception:
            pass
        
        # Method 2.2: Try to get gateway from /proc/net/route
        try:
            with open("/proc/net/route", "r") as f:
                for line in f:
                    parts = line.split()
                    if len(parts) >= 2 and parts[1] == "00000000":  # default route
                        if len(parts) >= 3:
                            gateway_hex = parts[2]
                            gateway_ip = ".".join([
                                str(int(gateway_hex[i:i+2], 16)) 
                                for i in range(6, -1, -2)
                            ])
                            try:
                                socket.inet_aton(gateway_ip)
                                # Gateway IP might not be external IP, but at least not container IP
                                if not any(gateway_ip.startswith(prefix) for prefix in container_ip_ranges):
                                    # If gateway IP is not in container IP range, can be used as fallback
                                    # But we still prefer IP returned by improved get_local_ip
                                    pass
                            except:
                                pass
        except Exception:
            pass
    
    # Return detected IP (get_local_ip already prefers returning non-container IP)
    return local_ip


class Settings(BaseSettings):
    """Application configuration"""
    
    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # MQTT configuration
    MQTT_ENABLED: bool = True  # Whether to enable MQTT service
    MQTT_USE_BUILTIN_BROKER: bool = True  # Whether to use built-in Broker (default use built-in)
    MQTT_BROKER: str = ""  # External Broker address (used when MQTT_USE_BUILTIN_BROKER=False, empty means auto use local IP)
    MQTT_BROKER_HOST: str = ""  # MQTT Broker external address (for Docker environment, empty means auto detect)
    MQTT_PORT: int = 1883  # MQTT port
    MQTT_BUILTIN_PORT: int = 1883  # Built-in Broker port
    MQTT_USERNAME: str = ""  # External Broker authentication (built-in Broker doesn't support auth yet)
    MQTT_PASSWORD: str = ""
    MQTT_UPLOAD_TOPIC: str = "annotator/upload/+"
    MQTT_RESPONSE_TOPIC_PREFIX: str = "annotator/response"
    MQTT_QOS: int = 1
    
    # Database configuration
    DATABASE_URL: str = f"sqlite:///{BASE_DIR}/data/annotator.db"
    
    # File storage configuration
    DATASETS_ROOT: Path = DATASETS_DIR
    MAX_IMAGE_SIZE_MB: int = 10
    
    # NE301 model compilation configuration
    NE301_PROJECT_PATH: str = ""  # NE301 project path (empty means use default path)
    NE301_USE_DOCKER: bool = True  # Whether to use Docker for compilation (default True)
    NE301_DOCKER_IMAGE: str = "camthink/ne301-dev:latest"  # Docker image name
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

# If MQTT_BROKER is empty, use local IP
if not settings.MQTT_BROKER:
    settings.MQTT_BROKER = get_local_ip()

# Ensure necessary directories exist
settings.DATASETS_ROOT.mkdir(parents=True, exist_ok=True)
(BASE_DIR / "data").mkdir(parents=True, exist_ok=True)

