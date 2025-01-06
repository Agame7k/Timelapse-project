import discord 
from discord import app_commands
import logging
import cv2
import io
import glob
import os
from datetime import datetime
from typing import Optional

from ..config.settings import Config 
from ..core.camera import Camera
from ..services.timelapse import TimelapseGenerator

class CameraCommands:
    """Camera-related commands"""
    
    def __init__(self, bot):
        self.bot = bot

    @app_commands.command(name="capture", description="Take images from all cameras")
    async def capture(self, interaction: discord.Interaction):
        """Capture images from all cameras"""
        if not await self._validate_channel(interaction):
            return

        if self.bot.maintenance_mode:
            await interaction.response.send_message(
                embed=discord.Embed(
                    title=f"{self.bot.EMOJIS['warning']} System in Maintenance",
                    description="Cannot capture while in maintenance mode.",
                    color=discord.Color.orange()
                )
            )
            return

        await interaction.response.defer()
        await self._execute_capture(interaction)

    @app_commands.command(name="timelapse", description="Generate timelapse video")
    async def timelapse(self, interaction: discord.Interaction, camera_index: int):
        """Generate a timelapse video for specified camera"""
        if not await self._validate_channel(interaction):
            return
            
        await interaction.response.defer()
        await self._generate_timelapse(interaction, camera_index)

    async def _execute_capture(self, interaction: discord.Interaction):
        """Execute capture operation with rich feedback"""
        # Implementation from original capture_command
        # ...

    async def _generate_timelapse(self, interaction: discord.Interaction, camera_index: int):
        """Generate timelapse with progress tracking"""
        # Implementation from original timelapse_command
        # ...

class StorageCommands:
    """Storage-related commands"""
    
    def __init__(self, bot):
        self.bot = bot

    @app_commands.command(name="upload", description="Upload pending images")
    async def upload(self, interaction: discord.Interaction):
        """Upload images to network storage"""
        if not await self._validate_channel(interaction):
            return
            
        await interaction.response.defer()
        await self._execute_upload(interaction)

    @app_commands.command(name="pull", description="Download from network storage") 
    async def pull(self, interaction: discord.Interaction):
        """Pull images from network storage"""
        if not await self._validate_channel(interaction):
            return
            
        await interaction.response.defer()
        await self._execute_pull(interaction)

class SystemCommands:
    """System management commands"""

    def __init__(self, bot):
        self.bot = bot

    @app_commands.command(name="status", description="View system status")
    async def status(self, interaction: discord.Interaction):
        """Show detailed system status"""
        if not await self._validate_channel(interaction):
            return
            
        await interaction.response.defer()
        await self._show_status(interaction)

    @app_commands.command(name="config", description="View configuration")
    async def config(self, interaction: discord.Interaction):
        """Show system configuration"""
        if not await self._validate_channel(interaction):
            return
            
        await self._show_config(interaction)

    @app_commands.command(name="maintenance", description="Toggle maintenance mode")
    @app_commands.default_permissions(administrator=True)
    async def maintenance(self, interaction: discord.Interaction):
        """Toggle system maintenance mode"""
        if not await self._validate_channel(interaction):
            return
            
        await self._toggle_maintenance(interaction)

    @app_commands.command(name="notifications", description="Toggle motion alerts")
    async def notifications(self, interaction: discord.Interaction):
        """Toggle motion detection notifications"""
        if not await self._validate_channel(interaction):
            return
            
        await self._toggle_notifications(interaction)

class AdminCommands:
    """Administrative commands"""

    def __init__(self, bot):
        self.bot = bot

    @app_commands.command(name="kill", description="Shut down the bot")
    @app_commands.default_permissions(administrator=True)
    async def kill(self, interaction: discord.Interaction):
        """Shut down the bot and services"""
        if not await self._validate_channel(interaction):
            return
            
        await self._execute_shutdown(interaction)

    @app_commands.command(name="reboot", description="Reboot system (Linux only)")
    @app_commands.default_permissions(administrator=True)
    async def reboot(self, interaction: discord.Interaction):
        """Reboot the entire system"""
        if not await self._validate_channel(interaction):
            return
            
        await self._execute_reboot(interaction)

def setup_commands(bot):
    """Register all command handlers"""
    camera_commands = CameraCommands(bot)
    storage_commands = StorageCommands(bot)
    system_commands = SystemCommands(bot)
    admin_commands = AdminCommands(bot)
    
    # Add commands to the bot's tree
    for cmd in [camera_commands, storage_commands, system_commands, admin_commands]:
        for command in cmd.__class__.__dict__.values():
            if isinstance(command, app_commands.Command):
                bot.tree.add_command(command)