import cv2
import logging
import threading
import asyncio
import time
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List, Dict, Any

from ..config.settings import CameraSettings
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class Camera:
    """
    Camera management class with motion detection and image capture capabilities.
    """
    def __init__(self, camera_index: int, settings: CameraSettings):
        """Initialize camera with specified index and settings."""
        self.camera_index = camera_index
        self.settings = settings
        
        # Camera state
        self.cap: Optional[cv2.VideoCapture] = None
        self.last_frame = None
        self.is_monitoring = False
        self._active = False
        self._initializing = False
        self._shutting_down = False
        
        # Motion detection state
        self.motion_value = 0.0
        self.motion_buffer: List[float] = []
        self.consecutive_motion = 0
        self.last_motion_alert = 0
        
        # Thread management
        self._cap_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None
        
        # Cooldown tracking
        self.last_capture = 0
        self.last_notification = 0
        
        logger.info(f"Camera {camera_index} instance created")

    async def initialize(self) -> bool:
        """
        Initialize camera hardware and configure settings.
        Returns True if successful, False otherwise.
        """
        try:
            with self._state_lock:
                if self._initializing or self._active:
                    return False
                    
                self._initializing = True
                
                # Release existing capture if any
                if self.cap:
                    self.cap.release()
                    
                # Initialize capture
                self.cap = cv2.VideoCapture(self.camera_index)
                if not self.cap.isOpened():
                    logger.error(f"Failed to open camera {self.camera_index}")
                    return False
                
                # Apply settings with fallback resolutions
                for resolution in self.settings.resolution_options:
                    if await self._configure_camera(resolution):
                        self._active = True
                        logger.info(f"Camera {self.camera_index} initialized at {resolution}")
                        return True
                        
                return False
                
        except Exception as e:
            logger.error(f"Camera {self.camera_index} initialization error: {e}")
            return False
        finally:
            self._initializing = False

    async def _configure_camera(self, resolution: Tuple[int, int]) -> bool:
        """Apply camera settings with specified resolution."""
        try:
            with self._cap_lock:
                # Set resolution
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
                
                # Apply other settings
                for prop, value in self.settings.default_settings.items():
                    if prop != 'resolution':
                        self.cap.set(getattr(cv2, f'CAP_PROP_{prop.upper()}'), value)
                
                # Verify settings
                actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                if abs(actual_width - resolution[0]) > 50 or \
                   abs(actual_height - resolution[1]) > 50:
                    return False
                
                # Test capture
                ret, frame = self.cap.read()
                return ret and frame is not None
                
        except Exception as e:
            logger.error(f"Camera {self.camera_index} configuration error: {e}")
            return False