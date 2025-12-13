"""Built-in MQTT Broker service"""
import asyncio
import logging
import threading
import socket
from typing import Optional
from backend.config import settings, get_local_ip

logger = logging.getLogger(__name__)


class BuiltinMQTTBroker:
    """Built-in MQTT Broker (using aMQTT)"""
    
    def __init__(self):
        self.broker = None
        self.is_running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
    
    def _run_broker(self):
        """Run Broker in separate thread"""
        try:
            from amqtt.broker import Broker
            
            # Create new event loop
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            
            async def start_broker():
                # Create Broker configuration (aMQTT format)
                config = {
                    'listeners': {
                        'default': {
                            'type': 'tcp',
                            'bind': '0.0.0.0',  # Bind all interfaces, allow connections from inside and outside container
                            'port': settings.MQTT_BUILTIN_PORT,
                            'max-connections': 100,  # Increase max connections
                        },
                    },
                    'sys_interval': 10,
                    'auth': {
                        'allow-anonymous': True,  # Allow anonymous connections
                    },
                    'keepalive-timeout': 300,  # Increase keepalive timeout (seconds)
                }
                
                # Create and start Broker
                self.broker = Broker(config)
                await self.broker.start()
                self.is_running = True
                
                local_ip = get_local_ip()
                logger.info(f"[MQTT Broker] Built-in MQTT Broker started on port {settings.MQTT_BUILTIN_PORT}")
                print(f"[MQTT Broker] Built-in MQTT Broker is ready at {local_ip}:{settings.MQTT_BUILTIN_PORT}")
                
                # Keep running
                try:
                    while self.is_running:
                        await asyncio.sleep(1)
                except asyncio.CancelledError:
                    pass
                finally:
                    if self.broker:
                        try:
                            await self.broker.shutdown()
                        except:
                            pass
            
            # Run async function
            self._loop.run_until_complete(start_broker())
            
        except ImportError as e:
            logger.error(f"[MQTT Broker] aMQTT library not installed. Please install: pip install amqtt. Error: {e}")
            self.is_running = False
        except Exception as e:
            logger.error(f"[MQTT Broker] Failed to start built-in broker: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.is_running = False
    
    def start(self):
        """Start built-in MQTT Broker (synchronous method)"""
        if self.is_running:
            logger.warning("[MQTT Broker] Broker is already running")
            return
        
        try:
            # Start Broker in separate thread
            self._thread = threading.Thread(target=self._run_broker, daemon=True)
            self._thread.start()
            
            # Wait a bit to ensure Broker starts
            import time
            time.sleep(0.5)
            
            if not self.is_running:
                raise RuntimeError("Failed to start built-in MQTT Broker")
                
        except Exception as e:
            logger.error(f"[MQTT Broker] Error starting broker thread: {e}")
            self.is_running = False
            raise
    
    def stop(self):
        """Stop built-in MQTT Broker"""
        if not self.is_running:
            return
        
        try:
            self.is_running = False
            
            if self._loop and self._loop.is_running():
                # Stop event loop
                self._loop.call_soon_threadsafe(self._loop.stop)
            
            if self._thread and self._thread.is_alive():
                self._thread.join(timeout=3)
            
            logger.info("[MQTT Broker] Built-in MQTT Broker stopped")
        except Exception as e:
            logger.error(f"[MQTT Broker] Error stopping broker: {e}")
    
    def get_broker_address(self) -> str:
        """Get Broker address"""
        local_ip = get_local_ip()
        return f"{local_ip}:{settings.MQTT_BUILTIN_PORT}"


# Global built-in Broker instance
builtin_mqtt_broker = BuiltinMQTTBroker()
