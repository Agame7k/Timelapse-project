import os
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import logging
from dotenv import load_dotenv

@dataclass
class CameraSettings:
    """Camera configuration settings."""
    resolution_options: List[Tuple[int, int]] = (
        (1280, 720),   # HD
        (1920, 1080),  # Full HD
        (2560, 1440),  # 2K
        (3840, 2160)   # 4K
    )
    
    default_resolution: Tuple[int, int] = (1280, 720)
    default_fps: int = 30
    default_brightness: int = 50
    default_contrast: int = 50
    default_saturation: int = 60
    default_exposure: int = -6
    default_gain: int = 0
    
    parameter_ranges: dict = {
        'brightness': (0, 100),
        'contrast': (-50, 50),
        'saturation': (0, 100),
        'exposure': (-10, 0),
        'gain': (0, 100)
    }

@dataclass
class StorageSettings:
    """Storage configuration settings."""
    primary_path: str = "timelapse_photos_primary/"
    backup_path: str = "timelapse_photos_backup/"
    
    def get_camera_path(self, base_path: str, camera_index: int) -> str:
        """Get camera-specific directory path."""
        return os.path.join(base_path, f"camera{camera_index}")

class Config:
    """Main configuration class for the timelapse application."""
    def __init__(self):
        self._load_environment()
        self._initialize_settings()
        self._validate_settings()

    def _load_environment(self) -> None:
        """Load environment variables from cred.env file."""
        env_path = Path('cred.env')
        if not env_path.exists():
            raise FileNotFoundError("cred.env file not found. Please create it using the template.")
        load_dotenv(dotenv_path=env_path)

    def _initialize_settings(self) -> None:
        """Initialize all configuration settings."""
        # Core settings
        self.COOLDOWN_SECONDS: int = 30
        self.MOTION_THRESHOLD: float = 1.0
        self.CAMERA_INDEXES: List[int] = [0, 2]
        self.FPS: int = 24
        
        # Motion detection settings
        self.MIN_MOTION_AREA: int = 500
        self.BLUR_SIZE: int = 21
        self.DILATE_ITERATIONS: int = 2
        self.MOTION_SENSITIVITY: int = 25
        
        # Load credentials from environment
        self.SMB_HOST = os.getenv('SMB_HOST', '')
        self.SMB_USERNAME = os.getenv('SMB_USERNAME', '')
        self.SMB_PASSWORD = os.getenv('SMB_PASSWORD', '')
        self.SMB_SHARE = os.getenv('SMB_SHARE', '')
        self.SMB_PATH = os.getenv('SMB_PATH', '/')
        self.DISCORD_TOKEN = os.getenv('DISCORD_TOKEN', '')

        # Convert Discord IDs
        try:
            self.GUILD_ID = int(os.getenv('GUILD_ID', '0'))
            self.COMMAND_CHANNEL_ID = int(os.getenv('COMMAND_CHANNEL_ID', '0'))
            self.MOTION_ALERT_CHANNEL_ID = int(os.getenv('MOTION_ALERT_CHANNEL_ID', '0'))
        except ValueError:
            raise ValueError("GUILD_ID, COMMAND_CHANNEL_ID, and MOTION_ALERT_CHANNEL_ID must be valid integers")

        # Alert settings
        self.MOTION_ALERT_COOLDOWN: int = 5  # minutes

    def _validate_settings(self) -> None:
        """Validate required configuration settings."""
        missing = []
        if not self.SMB_HOST:
            missing.append('SMB_HOST')
        if not self.GUILD_ID:
            missing.append('GUILD_ID')
        if not self.COMMAND_CHANNEL_ID:
            missing.append('COMMAND_CHANNEL_ID')

        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

    @property
    def camera_settings(self) -> CameraSettings:
        """Get camera configuration settings."""
        return CameraSettings()

    @property
    def storage_settings(self) -> StorageSettings:
        """Get storage configuration settings."""
        return StorageSettings()

# Global config instance
config = Config()