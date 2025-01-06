import os
import logging
import asyncio
import discord
from discord import app_commands
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import List, Optional

from ..config.settings import Config
from ..core.camera import Camera
from ..services.storage import StorageManager
from ..services.timelapse import TimelapseGenerator
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class DiscordBot(discord.Client):
    """Discord bot for managing timelapse camera system"""

    def __init__(self, config: Config):
        # Initialize with required intents
        intents = discord.Intents.default()
        intents.message_content = True
        
        super().__init__(intents=intents)
        
        self.config = config
        self.tree = app_commands.CommandTree(self)
        self.cameras: List[Camera] = []
        self.storage: Optional[StorageManager] = None
        
        # State tracking
        self.maintenance_mode = False
        self.start_time = datetime.now()
        self.shutting_down = False
        self.shutdown_event = asyncio.Event()
        self._ready = asyncio.Event()
        
        # Emoji constants for rich message formatting
        self.EMOJIS = {
            'success': '✅', 'error': '❌', 'warning': '⚠️',
            'camera': '📸', 'video': '🎥', 'upload': '📤',
            'status': '📊', 'config': '⚙️', 'motion': '🔄',
            'connected': '🟢', 'disconnected': '🔴', 
            'storage': '💾', 'download': '📥',
            'maintenance': '🔧', 'active': '⚡'
        }

    async def setup_hook(self) -> None:
        """Initialize bot services and commands"""
        await self._register_commands()
        await self._initialize_services()

    async def _initialize_services(self):
        """Initialize core services"""
        try:
            logger.info("Initializing bot services...")
            self.storage = StorageManager(self.config.storage_settings)
            self.cameras = await self._initialize_cameras()
            logger.info("Bot services initialized successfully")
        except Exception as e:
            logger.error(f"Service initialization failed: {e}")
            raise

    async def _initialize_cameras(self) -> List[Camera]:
        """Initialize camera subsystems"""
        cameras = []
        try:
            for idx in self.config.CAMERA_INDEXES:
                camera = Camera(idx, self.config.camera_settings)
                if await camera.initialize():
                    cameras.append(camera)
                    logger.info(f"Camera {idx} initialized")
        except Exception as e:
            logger.error(f"Camera initialization error: {e}")
            # Cleanup any initialized cameras
            for camera in cameras:
                await camera.deinitialize()
            raise
        return cameras

    async def _register_commands(self):
        """Register bot command tree"""
        try:
            await self.tree.sync()
            logger.info("Command tree synchronized")
        except Exception as e:
            logger.error(f"Command registration failed: {e}")
            raise

    async def start_bot(self):
        """Start the bot with the configured token"""
        try:
            async with self:
                await self.start(self.config.DISCORD_TOKEN)
        except Exception as e:
            logger.error(f"Bot startup failed: {e}")
            raise

    async def close(self):
        """Gracefully shut down the bot and services"""
        logger.info("Shutting down bot...")
        self.shutting_down = True
        self.shutdown_event.set()
        
        # Stop all cameras
        if self.cameras:
            for camera in self.cameras:
                await camera.deinitialize()
                
        # Close storage connections
        if self.storage:
            await self.storage.close()
            
        await super().close()
        logger.info("Bot shutdown complete")