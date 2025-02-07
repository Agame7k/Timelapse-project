# main.py
import os
import time as time_module
from datetime import datetime, timedelta, time
import logging
from dotenv import load_dotenv
import discord
from discord import app_commands
from discord.ext import tasks
import asyncio
import cv2
from smb.SMBConnection import SMBConnection
from PIL import Image
import io
import sys
import glob
import json
from pathlib import Path
import RPi.GPIO as GPIO
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from time import sleep
import threading
from pydub import AudioSegment
import numpy as np
import seaborn as sns
from sklearn.cluster import DBSCAN
from collections import Counter




BUZZER_PIN = 18  # GPIO pin for buzzer
BUZZER_ENABLED = True  # Control flag


# Load environment variables
load_dotenv('cred.env')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('timelapse.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TimelapseCamera:
    def __init__(self):
        # Initialize variables first
        self.notifications_enabled = True  # Fix the property name
        self.is_monitoring = False
        self.is_ready = False
        self.is_fully_initialized = False
        self.start_time = time_module.time()
        self.last_capture_time = 0
        self.capture_interval = 60
        self.notifications_enabled = True
        self.timesheet_data = {}
        self.active_clock = None
        self.owner_id = int(os.getenv('OWNER_ID'))  # Add OWNER_ID to your cred.env
        self.timesheet_file = "timesheet.json"
        self.heatmap = np.zeros((480, 640))  # Adjust size to match your camera resolution
        self.heatmap_start_time = datetime.now()
        self.heatmap_cooldown = 0.5  # Seconds between heatmap updates
        self.load_timesheet()
        
        # Discord initialization
        self.discord_client = discord.Client(intents=discord.Intents.all())
        self.tree = app_commands.CommandTree(self.discord_client)
        
        # Camera initialization
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                logger.error("Failed to open camera")
                raise RuntimeError("Could not initialize camera")
                
            ret, _ = self.camera.read()
            if not ret:
                self.camera.release()
                logger.error("Camera read test failed")
                raise RuntimeError("Camera read test failed")
                
            logger.info("Camera initialized successfully")
        except Exception as e:
            logger.error(f"Camera initialization failed: {e}")
            raise
            
        # Initialize other variables
        self.motion_threshold = 25
        self.min_area = 500
        self.background = None
        self.last_motion_time = 0
        self.motion_cooldown = 30
        self.discord_channel = None
        
        # SMB Configuration
        try:
            self.smb_client = SMBConnection(
                os.getenv('SMB_USERNAME'),
                os.getenv('SMB_PASSWORD'),
                os.getenv('CLIENT_NAME'),
                os.getenv('SERVER_NAME'),
                use_ntlm_v2=True
            )
            logger.info("SMB client configured")
        except Exception as e:
            logger.error(f"SMB configuration failed: {e}")
            raise
        
        # Set up Discord
        self.setup_discord()


    def play_buzzer_tone(self, frequency, duration):
        """Play a single tone on the buzzer with proper cleanup"""
        if not BUZZER_ENABLED:
            return
            
        try:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(BUZZER_PIN, GPIO.OUT)
            buzzer = GPIO.PWM(BUZZER_PIN, frequency)
            buzzer.start(50)
            sleep(duration)
            buzzer.stop()
            GPIO.cleanup(BUZZER_PIN)  # Cleanup after each tone
        except Exception as e:
            logger.error(f"Error playing tone: {e}")

    async def update_heatmap(self, contours):
        """Update motion heatmap with new contours"""
        try:
            mask = np.zeros_like(self.heatmap)
            for contour in contours:
                cv2.drawContours(mask, [contour], -1, (1), -1)
            self.heatmap += mask
            logger.debug("Heatmap updated with new motion data")
        except Exception as e:
            logger.error(f"Error updating heatmap: {e}")

    def load_timesheet(self):
        """Load timesheet data from JSON file and restore active session if exists"""
        try:
            if os.path.exists(self.timesheet_file):
                with open(self.timesheet_file, 'r') as f:
                    self.timesheet_data = json.load(f)
                    # Check for and restore active session
                    if "active_session" in self.timesheet_data:
                        session = self.timesheet_data["active_session"]
                        if session:
                            self.active_clock = {
                                "start_time": datetime.strptime(session["start_time"], "%Y-%m-%d %H:%M:%S"),
                                "auto_checkout_time": datetime.strptime(session["auto_checkout_time"], "%Y-%m-%d %H:%M:%S")
                            }
                    else:
                        self.timesheet_data["active_session"] = None
            else:
                self.timesheet_data = {
                    "entries": [], 
                    "total_hours": 0,
                    "active_session": None
                }
        except Exception as e:
            logger.error(f"Failed to load timesheet: {str(e)}")
            self.timesheet_data = {
                "entries": [], 
                "total_hours": 0,
                "active_session": None
            }

    def save_timesheet(self):
        """Save timesheet data to JSON file"""
        try:
            with open(self.timesheet_file, 'w') as f:
                json.dump(self.timesheet_data, f, indent=4)
            # Upload to SMB
            asyncio.create_task(self.upload_to_smb(self.timesheet_file))
        except Exception as e:
            logger.error(f"Failed to save timesheet: {str(e)}")

    @tasks.loop(time=time(hour=20, minute=28)) #20:28 UTC = 2:28 PM CST
    async def reminder_task(self):
        """Send daily reminder at 1:25 PM on weekdays"""
        if datetime.now().weekday() < 5:  # 0-4 are Monday to Friday
            try:
                owner = await self.discord_client.fetch_user(self.owner_id)
                if owner:
                    embed = discord.Embed(
                        title="⏰ Clock-In Reminder",
                        description="Don't forget to clock in! Use `/start_clock` to begin tracking time.",
                        color=discord.Color.blue()
                    )
                    await owner.send(embed=embed)
            except Exception as e:
                logger.error(f"Failed to send reminder: {str(e)}")

    async def initialize(self):
        """Initialize the system fully"""
        try:
            # Wait for Discord to be ready
            while not self.discord_client.is_ready():
                await asyncio.sleep(1)
            
            # Set initial status
            self.is_ready = True
            self.is_monitoring = True
            self.is_fully_initialized = True
            
            logger.info("System fully initialized")
            self.reminder_task.start()
        
            return True
        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False
       
    def setup_discord(self):
        """Initialize Discord bot and commands"""
        @self.discord_client.event

        async def on_ready():
            try:
                await self.tree.sync()
                logger.info("Command tree synchronized successfully")
                
                # Set up channel
                channel_id = os.getenv('DISCORD_CHANNEL_ID')
                if not channel_id:
                    logger.error("DISCORD_CHANNEL_ID not found in environment variables")
                    return
                    
                self.discord_channel = self.discord_client.get_channel(int(channel_id))
                if not self.discord_channel:
                    logger.error(f"Could not find Discord channel with ID {channel_id}")
                    return
                
                activity = discord.Activity(
                    type=discord.ActivityType.watching,
                    name="ROBOTS"
                )
                await self.discord_client.change_presence(status=discord.Status.dnd, activity=activity)
                logger.info(f'Discord bot logged in as {self.discord_client.user}')
                startup_embed = discord.Embed(
                title="🤖 Bot Online",
                description="Timelapse monitoring system is now operational",
                color=discord.Color.brand_green(),
                timestamp=datetime.now()
            )

                startup_embed.add_field(
                    name="🔧 System Info",
                    value=f"""```
                Python v{sys.version.split()[0]}
                Discord.py v{discord.__version__}
                OpenCV v{cv2.__version__}
                ```""",
                    inline=False
                )

                startup_embed.add_field(
                    name="📊 Status",
                    value=f"```\n"
                        f"Monitoring: {'Active' if self.is_monitoring else 'Paused'}\n"
                        f"Camera: {'Connected' if self.camera.isOpened() else 'Error'}\n"
                        f"Notifications: {'Enabled' if self.notifications_enabled else 'Disabled'}\n"
                        f"```",
                    inline=True
                )

                startup_embed.add_field(
                    name="⚙️ Settings",
                    value=f"```\n"
                        f"Motion Threshold: {self.motion_threshold}\n"
                        f"Min Area: {self.min_area}px\n"
                        f"Cooldown: {self.motion_cooldown}s\n"
                        f"```",
                    inline=True
                )

                startup_embed.set_footer(text="Use /status for detailed system information")

                try:
                    await self.discord_channel.send(embed=startup_embed)
                    logger.info("Startup message sent to Discord")
                except Exception as e:
                    logger.error(f"Failed to send startup message: {str(e)}")
                    self.is_ready = True
                    self.is_fully_initialized = True  # Set initialization flag
            except Exception as e:
                logger.error(f"Error during bot startup: {str(e)}")

        def create_progress_bar(value, max_value, length=10):
            """Create a progress bar with custom length"""
            filled = int(value / max_value * length)
            bar = '█' * filled + '░' * (length - filled)
            return bar

        def get_folder_size(pattern):
            """Get total size of folders matching pattern in bytes"""
            total = 0
            for folder in glob.glob(pattern):
                if os.path.isdir(folder):
                    for path, dirs, files in os.walk(folder):
                        for f in files:
                            total += os.path.getsize(os.path.join(path, f))
            return total


        @self.tree.command(name="toggle-motion", description="Toggle motion monitoring")
        async def toggle_monitoring(interaction: discord.Interaction):
            self.is_monitoring = not self.is_monitoring
            status = "Enabled" if self.is_monitoring else "Disabled"
            
            # Update Discord status with DND when monitoring, Idle when not
            await self.discord_client.change_presence(
                status=discord.Status.dnd if self.is_monitoring else discord.Status.idle,
                activity=discord.Activity(
                    type=discord.ActivityType.watching if self.is_monitoring else discord.ActivityType.listening,
                    name="ROBOTS" if self.is_monitoring else "cause im blind"
                )
            )
            
            embed = discord.Embed(
                title="Motion Monitoring",
                description=f"System monitoring has been {status.lower()}",
                color=discord.Color.green() if self.is_monitoring else discord.Color.orange(),
                timestamp=datetime.now()
            )
            embed.add_field(
                name="Status", 
                value="🟢 Active" if self.is_monitoring else "🟠 Paused",
                inline=True
            )
            embed.add_field(
                name="Triggered By",
                value=f"👤 {interaction.user.mention}",
                inline=True
            )
            embed.set_footer(text="Use /status to check system status")
            
            await interaction.response.send_message(embed=embed)
            logger.info(f"Motion monitoring {status.lower()} by {interaction.user}")

        @self.tree.command(name="status", description="Get detailed system status")
        async def status(interaction: discord.Interaction):
            await interaction.response.defer()
            
            try:
                import psutil
                import platform
                
                # System Hardware Stats
                cpu_freq = psutil.cpu_freq()
                cpu_count = psutil.cpu_count()
                cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                temps = psutil.sensors_temperatures() if hasattr(psutil, 'sensors_temperatures') else {}
                network = psutil.net_io_counters()
                boot_time = datetime.fromtimestamp(psutil.boot_time())
                
                # Image Stats
                captures_today = len(glob.glob(f"captures_{datetime.now().strftime('%Y%m%d')}/*.jpg"))
                total_captures = sum(len(glob.glob(f"{f}/*.jpg")) for f in glob.glob('captures_*'))
                timelapse_photos = len(glob.glob('timelapse_photos_primary/*.jpg'))
                
                # Calculate uptime
                uptime = time_module.time() - self.start_time
                uptime_str = f"{int(uptime // 3600)}h {int((uptime % 3600) // 60)}m {int(uptime % 60)}s"
                
                def format_size(size):
                    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
                        if size < 1024.0:
                            return f"{size:.1f}{unit}"
                        size /= 1024.0
                
                # Main embed with system overview
                embed = discord.Embed(
                    title="System Status Dashboard",
                    description=(
                        f"**System Health Overview**\n"
                        f"🟢 System Online | "
                        f"💻 {cpu_percent[0]:.1f}% CPU | "
                        f"🔧 {memory.percent}% RAM | "
                        f"💾 {disk.percent}% Disk"
                    ),
                    color=discord.Color.brand_green(),
                    timestamp=datetime.now()
)

                # System Information
                sys_info = (
                    f"```ml\n"
                    f"Hostname   : {platform.node()}\n"
                    f"OS         : {platform.system()} {platform.release()}\n"
                    f"Boot Time  : {boot_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                    f"Uptime     : {uptime_str}\n"
                    f"Python     : v{platform.python_version()}\n"
                    f"OpenCV     : v{cv2.__version__}\n"
                    f"Discord.py : v{discord.__version__}\n"
                    f"```"
                )
                embed.add_field(name="🖥️ System Information", value=sys_info, inline=False)

                # Resource Metrics
                metrics = (
                    f"```ml\n"
                    f"CPU Usage    : {create_progress_bar(cpu_percent[0], 100, 15)} {cpu_percent[0]:>5.1f}%\n"
                    f"Memory Usage : {create_progress_bar(memory.percent, 100, 15)} {memory.percent:>5.1f}%\n"
                    f"Disk Usage   : {create_progress_bar(disk.percent, 100, 15)} {disk.percent:>5.1f}%\n"
                    f"```"
                )
                embed.add_field(name="📊 Resource Metrics", value=metrics, inline=False)

                # Hardware Details
                hw_info = (
                    f"```ml\n"
                    f"CPU Model : {platform.processor()[:40]}...\n"
                    f"Cores     : {cpu_count} ({psutil.cpu_count(logical=False)} physical)\n"
                    f"Memory    : {format_size(memory.total)} ({format_size(memory.used)} used)\n"
                    f"Disk      : {format_size(disk.total)} ({format_size(disk.free)} free)\n"
                    f"Network   : ↓ {format_size(network.bytes_recv)} | ↑ {format_size(network.bytes_sent)}\n"
                    f"```"
                )
                embed.add_field(name="💻 Hardware", value=hw_info, inline=False)

                # Camera Status
                motion_time = "Never" if self.last_motion_time == 0 else f"{int(time_module.time() - self.last_motion_time)}s ago"
                camera = (
                    f"```ml\n"
                    f"Status      : {'🟢 Connected' if self.camera.isOpened() else '🔴 Disconnected'}\n"
                    f"Mode        : {'📸 Active' if self.is_monitoring else '⏸️ Paused'}\n"
                    f"Last Motion : {motion_time}\n"
                    f"Settings    : {self.motion_threshold} threshold, {self.min_area}px min area\n"
                    f"Alerts      : {'🔔 Enabled' if self.notifications_enabled else '🔕 Disabled'}\n"
                    f"```"
                )
                embed.add_field(name="📸 Camera Status", value=camera, inline=False)

                # Storage Statistics
                storage_used = get_folder_size('captures_*')
                timelapse_used = get_folder_size('timelapse_photos_primary')
                stats = (
                    f"```ml\n"
                    f"Today's Captures : {captures_today:,} images\n"
                    f"Total Captures  : {total_captures:,} images\n"
                    f"Storage Used    : {format_size(storage_used + timelapse_used)}\n"
                    f"Capture Rate    : {captures_today / ((time_module.time() - self.start_time) / 3600):.1f} img/hour\n"
                    f"```"
                )
                embed.add_field(name="📁 Storage Statistics", value=stats, inline=False)

                # Temperature Monitoring (if available)
                if temps:
                    temp_info = "```ml\n"
                    for sensor_name, entries in temps.items():
                        for entry in entries:
                            status = '🟢' if entry.current < 60 else '🟡' if entry.current < 80 else '🔴'
                            temp_info += f"{sensor_name:<12}: {status} {entry.current:>5.1f}°C"
                            if entry.high:
                                temp_info += f" (max: {entry.high:>5.1f}°C)\n"
                    temp_info += "```"
                    embed.add_field(name="🌡️ Temperature", value=temp_info, inline=False)

                # Network Status
                net_status = (
                    f"```ml\n"
                    f"Discord : {'🟢 Connected' if self.discord_client.is_ready() else '🔴 Disconnected'}\n"
                    f"Latency : {round(self.discord_client.latency * 1000)}ms\n"
                    f"SMB     : {'🟢 Connected' if self.smb_client else '🔴 Disconnected'}\n"
                    f"```"
                )
                embed.add_field(name="🌐 Network Status", value=net_status, inline=False)

                # Add footer with refresh info
                embed.set_footer(
                    text=f"Last Updated: {datetime.now().strftime('%H:%M:%S')} • Refresh with /status",
                    icon_url="https://i.imgur.com/XwK0v9F.png"  # Optional: Add a small icon
                )

                await interaction.followup.send(embed=embed)
                logger.info(f"Status requested by {interaction.user}")

            except Exception as e:
                error_embed = discord.Embed(
                    title="❌ Status Error",
                    description=f"Failed to gather system information: {str(e)}",
                    color=discord.Color.red()
                )
                await interaction.followup.send(embed=error_embed)
                logger.error(f"Status command failed: {str(e)}")

        @self.tree.command(name="snapshot", description="Take and send a snapshot")
        async def snapshot(interaction: discord.Interaction):
            await interaction.response.defer()
            logger.info("Taking snapshot")
            start_time = time_module.time()
            
            ret, frame = self.camera.read()
            if not ret:
                embed = discord.Embed(
                    title="❌ Snapshot Failed",
                    description="Failed to capture image from camera",
                    color=discord.Color.red()
                )
                await interaction.followup.send(embed=embed)
                return

            success, buffer = cv2.imencode('.jpg', frame)
            if not success:
                embed = discord.Embed(
                    title="❌ Processing Failed",
                    description="Failed to process captured image",
                    color=discord.Color.red()
                )
                await interaction.followup.send(embed=embed)
                return

            io_buf = io.BytesIO(buffer)
            file = discord.File(fp=io_buf, filename='snapshot.jpg')
            
            embed = discord.Embed(
                title="📸 Snapshot Captured",
                color=discord.Color.blue(),
                timestamp=datetime.now()
            )
            embed.add_field(
                name="Timestamp",
                value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                inline=True
            )
            embed.add_field(
                name="Requested By",
                value=interaction.user.mention,
                inline=True
            )
            embed.add_field(
                name="Processing Time",
                value=f"{(time_module.time() - start_time):.2f}s",
                inline=True
            )
            embed.set_image(url="attachment://snapshot.jpg")
            
            await interaction.followup.send(embed=embed, file=file)

        @self.tree.command(name="trigger", description="Manually trigger motion detection")
        async def trigger_motion(interaction: discord.Interaction):
            await interaction.response.defer()
            
            if not self.is_monitoring:
                embed = discord.Embed(
                    title="❌ Trigger Failed",
                    description="Cannot trigger: Motion monitoring is paused",
                    color=discord.Color.red()
                )
                await interaction.followup.send(embed=embed)
                return
                
            ret, frame = self.camera.read()
            if not ret:
                embed = discord.Embed(
                    title="❌ Capture Failed",
                    description="Failed to capture frame from camera",
                    color=discord.Color.red()
                )
                await interaction.followup.send(embed=embed)
                return
                
            logger.info(f"Manual motion trigger by {interaction.user}")
            await self.capture_and_save(frame)
            
            embed = discord.Embed(
                title="🎯 Motion Triggered",
                description="Manual motion detection triggered successfully",
                color=discord.Color.green(),
                timestamp=datetime.now()
            )
            embed.add_field(
                name="Triggered By",
                value=interaction.user.mention,
                inline=True
            )
            embed.set_footer(text="Check above for the captured image")
            
            await interaction.followup.send(embed=embed)
            
        @self.tree.command(name="kill", description="Safely shutdown the bot and camera system")
        async def kill(interaction: discord.Interaction):
            if not interaction.user.guild_permissions.administrator:
                embed = discord.Embed(
                    title="❌ Permission Denied",
                    description="Only administrators can shut down the bot",
                    color=discord.Color.red()
                )
                await interaction.response.send_message(embed=embed, ephemeral=True)
                return

            # Initial confirmation embed
            confirm_embed = discord.Embed(
                title="⚠️ Shutdown Confirmation",
                description="Are you sure you want to shutdown the system?",
                color=discord.Color.yellow(),
                timestamp=datetime.now()
            )
            
            confirm_embed.add_field(
                name="System Status",
                value=f"""```
        Monitoring: {'Active' if self.is_monitoring else 'Paused'}
        Uptime: {self.format_uptime()}
        Captured Today: {len(glob.glob(f"captures_{datetime.now().strftime('%Y%m%d')}/*.jpg"))} images
        Last Motion: {self.format_last_motion_time()}```""",
                inline=False
            )
            
            # Create confirm/cancel buttons
            confirm_button = discord.ui.Button(
                style=discord.ButtonStyle.danger,
                label="Confirm Shutdown",
                emoji="⚡",
                custom_id="confirm_shutdown"
            )
            
            cancel_button = discord.ui.Button(
                style=discord.ButtonStyle.secondary,
                label="Cancel",
                emoji="✖️",
                custom_id="cancel_shutdown"
            )

            async def confirm_callback(interaction: discord.Interaction):
                for child in view.children:
                    child.disabled = True
                await interaction.response.edit_message(view=view)
                
                # Execute shutdown sequence
                await self.shutdown_sequence()

            async def cancel_callback(interaction: discord.Interaction):
                cancel_embed = discord.Embed(
                    title="✅ Shutdown Cancelled",
                    description="System will continue running normally",
                    color=discord.Color.green()
                )
                for child in view.children:
                    child.disabled = True
                await interaction.response.edit_message(embed=cancel_embed, view=view)

            # Set up button callbacks
            confirm_button.callback = confirm_callback
            cancel_button.callback = cancel_callback

            # Create view and add buttons
            view = discord.ui.View(timeout=30)
            view.add_item(confirm_button)
            view.add_item(cancel_button)

            # Send confirmation message
            await interaction.response.send_message(embed=confirm_embed, view=view)

            # Handle timeout
            async def on_timeout():
                timeout_embed = discord.Embed(
                    title="⏰ Shutdown Cancelled",
                    description="Confirmation timed out after 30 seconds",
                    color=discord.Color.greyple()
                )
                for child in view.children:
                    child.disabled = True
                await interaction.edit_original_response(embed=timeout_embed, view=view)

            view.on_timeout = on_timeout
            
        @self.tree.command(name="create_timelapse", description="Create timelapse from motion captures")
        async def create_timelapse(
            interaction: discord.Interaction,
            fps: int = 60,
            specific_date: str = None  # Changed from date to specific_date and made optional
        ):
            await interaction.response.defer()
            
            # Initial embed
            embed = discord.Embed(
                title="🎬 Creating Timelapse",
                description="Processing captured images...",
                color=discord.Color.blue(),
                timestamp=datetime.now()
            )
            embed.add_field(name="FPS", value=str(fps), inline=True)
            embed.add_field(
                name="Mode", 
                value=f"Single Day: {specific_date}" if specific_date else "All Captures",
                inline=True
            )
            
            progress_message = await interaction.followup.send(embed=embed)
            
            try:
                # Get all capture folders if no specific date
                if specific_date:
                    folders = [f"captures_{specific_date}"]
                else:
                    folders = sorted([f for f in glob.glob('captures_*') if os.path.isdir(f)])
                
                if not folders:
                    embed.title = "❌ Timelapse Failed"
                    embed.description = "No capture folders found"
                    embed.color = discord.Color.red()
                    await progress_message.edit(embed=embed)
                    return
                
                start_time = time_module.time()
                # Count total images across all folders
                total_images = 0
                all_images = []
                for folder in folders:
                    if os.path.exists(folder):
                        images = sorted([img for img in os.listdir(folder) if img.endswith(".jpg")])
                        total_images += len(images)
                        all_images.extend([os.path.join(folder, img) for img in images])
                
                if not total_images:
                    embed.title = "❌ Timelapse Failed"
                    embed.description = "No images found in capture folders"
                    embed.color = discord.Color.red()
                    await progress_message.edit(embed=embed)
                    return
                
                embed.add_field(name="Total Images", value=str(total_images), inline=True)
                await progress_message.edit(embed=embed)
                
                # Create video using first image dimensions
                first_image = cv2.imread(all_images[0])
                height, width, _ = first_image.shape
                video_path = "complete_timelapse.mp4" if not specific_date else f"captures_{specific_date}_timelapse.mp4"
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
                
                # Process images with progress bar
                for i, image_path in enumerate(all_images):
                    video.write(cv2.imread(image_path))
                    
                    if i % 10 == 0 or i == total_images - 1:  # Update every 10 frames or on last frame
                        # Calculate timing metrics
                        elapsed_time = time_module.time() - start_time
                        frames_processed = i + 1
                        processing_rate = frames_processed / elapsed_time if elapsed_time > 0 else 0
                        frames_remaining = total_images - frames_processed
                        eta = frames_remaining / processing_rate if processing_rate > 0 else 0
                        
                        # Calculate progress percentage
                        progress = frames_processed / total_images
                        
                        # Create progress bar
                        bar_length = 20
                        filled_length = int(bar_length * progress)
                        bar = '█' * filled_length + '░' * (bar_length - filled_length)
                        
                        # Format time strings
                        elapsed_str = f"{int(elapsed_time//60)}:{int(elapsed_time%60):02d}"
                        eta_str = f"{int(eta//60)}:{int(eta%60):02d}"
                        
                        # Update embed description
                        embed.description = (
                            f"Processing images... ({frames_processed}/{total_images})\n"
                            f"`{bar}` {progress*100:.1f}%\n"
                            f"Elapsed: {elapsed_str} | ETA: {eta_str}"
                        )
                        await progress_message.edit(embed=embed)
                
                video.release()
                logger.info(f"Created timelapse video: {video_path}")
                
                # Upload to SMB
                await self.upload_to_smb(video_path)
                
                # Create final embed with video
                success_embed = discord.Embed(
                    title="✅ Timelapse Created",
                    description="Video has been created and uploaded successfully",
                    color=discord.Color.green(),
                    timestamp=datetime.now()
                )
                success_embed.add_field(name="FPS", value=str(fps), inline=True)
                success_embed.add_field(name="Frame Count", value=str(total_images), inline=True)
                success_embed.add_field(name="Duration", value=f"{total_images/fps:.1f}s", inline=True)
                success_embed.add_field(
                    name="Time Range",
                    value=f"All captures" if not specific_date else f"Date: {specific_date}",
                    inline=False
                )
                
                # Attach video file if under Discord's file size limit (8MB for most servers)
                if os.path.getsize(video_path) < 8_000_000:  # 8MB in bytes
                    file = discord.File(video_path, filename="timelapse.mp4")
                    success_embed.add_field(
                        name="📹 Preview",
                        value="Video attached below",
                        inline=False
                    )
                    await progress_message.edit(embed=success_embed)
                    await interaction.followup.send(file=file)
                else:
                    success_embed.add_field(
                        name="📹 Video",
                        value=f"Video size exceeds Discord's limit. Access it at: `{video_path}` on The SMB server",
                        inline=False
                    )
                    await progress_message.edit(embed=success_embed)
                
            except Exception as e:
                error_embed = discord.Embed(
                    title="❌ Timelapse Failed",
                    description=f"Error: {str(e)}",
                    color=discord.Color.red(),
                    timestamp=datetime.now()
                )
                await progress_message.edit(embed=error_embed)
                logger.error(f"Failed to create timelapse: {str(e)}")

        @self.tree.command(name="reboot_pi", description="Reboot the Raspberry Pi system")
        async def reboot_pi(interaction: discord.Interaction):
            if not interaction.user.guild_permissions.administrator:
                embed = discord.Embed(
                    title="❌ Permission Denied",
                    description="Only administrators can reboot the system",
                    color=discord.Color.red()
                )
                await interaction.response.send_message(embed=embed, ephemeral=True)
                return
            
            embed = discord.Embed(
                title="🔄 System Reboot",
                description="Initiating Raspberry Pi reboot sequence...",
                color=discord.Color.orange(),
                timestamp=datetime.now()
            )
            embed.add_field(
                name="Triggered By",
                value=interaction.user.mention,
                inline=True
            )
            embed.add_field(
                name="Uptime",
                value=self.format_uptime(),
                inline=True
            )
            embed.set_footer(text="System will reboot momentarily")
            
            await interaction.response.send_message(embed=embed)
            logger.info(f"System reboot initiated by {interaction.user}")
            
            # Cleanup and execute reboot
            await self.discord_client.close()
            self.cleanup()
            os.system("sudo reboot")

        @self.tree.command(name="sync", description="Sync files between Pi and SMB server")
        async def sync_files(interaction: discord.Interaction):
            await interaction.response.defer()
            start_time = time_module.time()
            
            # Get share name from environment variables
            share_name = os.getenv('SMB_SHARE_NAME')
            if not share_name:
                error_embed = discord.Embed(
                    title="❌ Sync Failed",
                    description="SMB_SHARE_NAME not found in environment variables",
                    color=discord.Color.red(),
                    timestamp=datetime.now()
                )
                await interaction.followup.send(embed=error_embed)
                logger.error("SMB_SHARE_NAME not found in environment variables")
                return
            
            
            
            progress_embed = discord.Embed(
                title="🔄 Starting Sync",
                description="Preparing to sync files...",
                color=discord.Color.blue(),
                timestamp=datetime.now()
            )
            progress_message = await interaction.followup.send(embed=progress_embed)
            
            try:
                # Create new SMB connection for sync operation
                smb = SMBConnection(
                    os.getenv('SMB_USERNAME'),
                    os.getenv('SMB_PASSWORD'),
                    os.getenv('CLIENT_NAME'),
                    os.getenv('SERVER_NAME'),
                    use_ntlm_v2=True
                )

                if not smb.connect(
                    os.getenv('SMB_SERVER_IP'),
                    int(os.getenv('SMB_PORT', 445)),
                    timeout=30
                ):
                    raise Exception("Failed to connect to SMB server")

                # Verify share exists
                remote_shares = smb.listShares()
                if not any(share.name == share_name for share in remote_shares):
                    raise Exception(f"Share '{share_name}' not found on server")

                # Initialize stats
                stats = {'uploaded': 0, 'downloaded': 0, 'skipped': 0, 'errors': 0}
                total_files = 0
                processed_files = 0

                remote_files = {}
                try:
                    # Initialize remote_files dictionary
                    remote_files = {}
                    # Modify folder patterns to explicitly match directories and exclude mp4 files
                    folder_patterns = {
                        'captures_*': lambda x: os.path.isdir(x) and not x.endswith('.mp4'),
                        'timelapse_photos_primary': os.path.isdir
                    }
                    
                    # Process each folder pattern
                    for pattern, is_valid_dir in folder_patterns.items():
                        for current_dir in glob.glob(pattern):
                            if not is_valid_dir(current_dir):
                                logger.debug(f"Skipping non-directory: {current_dir}")
                                continue
                                
                            try:
                                # List files in remote directory
                                file_list = smb.listPath(share_name, current_dir)
                                for f in file_list:
                                    # Check if it's a file (not a directory)
                                    if not (f.file_attributes & 0x10):
                                        remote_files[f.filename] = f.last_write_time
                            except Exception as e:
                                logger.error(f"Failed to list files in {current_dir}: {str(e)}")
                                continue
                except Exception as e:
                    logger.error(f"Failed to list remote files: {str(e)}")
                    raise
                
                # Count total files first
                total_files = 0
                for pattern, is_valid_dir in folder_patterns.items():
                    for local_dir in glob.glob(pattern):
                        if is_valid_dir(local_dir):
                            total_files += len([f for f in os.listdir(local_dir) 
                                            if os.path.isfile(os.path.join(local_dir, f))])

                # Update progress message
                progress_embed.description = f"Found {total_files} files to process"
                await progress_message.edit(embed=progress_embed)
                
                # Process files with progress updates
                for pattern, is_valid_dir in folder_patterns.items():
                    for local_dir in glob.glob(pattern):
                        if not is_valid_dir(local_dir):
                            continue

                        try:
                            # Create remote directory if needed
                            try:
                                smb.listPath(share_name, local_dir)
                            except:
                                smb.createDirectory(share_name, local_dir)
                                logger.info(f"Created remote directory: {local_dir}")

                            # Get remote files list for current directory
                            remote_files = {}
                            try:
                                file_list = smb.listPath(share_name, local_dir)
                                for f in file_list:
                                    # Check if it's a file (not a directory)
                                    if not (f.file_attributes & 0x10):
                                        remote_files[f.filename] = f.last_write_time
                            except Exception as e:
                                logger.error(f"Failed to list remote files in {local_dir}: {str(e)}")
                                continue

                            # Get local files
                            local_files = {}
                            for f in os.listdir(local_dir):
                                if os.path.isfile(os.path.join(local_dir, f)):
                                    local_files[f] = os.path.getmtime(os.path.join(local_dir, f))
                            
                            # Process each file
                            update_interval = 5
                            last_update = time_module.time()
                            
                            for i, (fname, local_mtime) in enumerate(local_files.items()):
                                try:
                                    # Upload if file is new or modified
                                    if fname not in remote_files or local_mtime > remote_files[fname]:
                                        local_path = os.path.join(local_dir, fname)
                                        remote_path = f"{local_dir}/{fname}"
                                        
                                        with open(local_path, 'rb') as file:
                                            if smb.storeFile(share_name, remote_path, file):
                                                stats['uploaded'] += 1
                                                logger.info(f"Uploaded: {remote_path}")
                                    else:
                                        stats['skipped'] += 1
                                    
                                    processed_files += 1
                                    
                                    # Update progress
                                    if i % update_interval == 0 or (time_module.time() - last_update) > 2:
                                        progress = (processed_files / total_files) * 100
                                        elapsed_time = time_module.time() - start_time
                                        eta = (elapsed_time / processed_files) * (total_files - processed_files) if processed_files > 0 else 0
                                        
                                        # Create progress bar
                                        bar_length = 20
                                        filled_length = int(bar_length * processed_files // total_files)
                                        bar = '█' * filled_length + '░' * (bar_length - filled_length)
                                        
                                        progress_embed.description = (
                                            f"Progress: {processed_files}/{total_files} files\n"
                                            f"`{bar}` {progress:.1f}%\n"
                                            f"ETA: {int(eta/60)}m {int(eta%60)}s\n"
                                            f"```\n"
                                            f"Uploaded: {stats['uploaded']}\n"
                                            f"Skipped: {stats['skipped']}\n"
                                            f"Errors: {stats['errors']}\n"
                                            f"```"
                                        )
                                        await progress_message.edit(embed=progress_embed)
                                        last_update = time_module.time()
                                        
                                except Exception as e:
                                    stats['errors'] += 1
                                    logger.error(f"Upload failed for {fname}: {str(e)}")
                                    continue

                        except Exception as e:
                            logger.error(f"Error processing directory {local_dir}: {str(e)}")
                            continue

                # Create final summary embed
                elapsed_time = time_module.time() - start_time
                summary_embed = discord.Embed(
                    title="✅ Sync Complete" if stats['errors'] == 0 else "⚠️ Sync Completed with Errors",
                    description=(
                        f"Processed {total_files} files in {elapsed_time:.1f}s\n\n"
                        f"```\n"
                        f"📤 Uploaded: {stats['uploaded']}\n"
                        f"⏭️ Skipped: {stats['skipped']}\n"
                        f"❌ Errors: {stats['errors']}\n"
                        f"⏱️ Time: {int(elapsed_time/60)}m {int(elapsed_time%60)}s\n"
                        f"```"
                    ),
                    color=discord.Color.green() if stats['errors'] == 0 else discord.Color.yellow(),
                    timestamp=datetime.now()
                )
                
                await progress_message.edit(embed=summary_embed)
                logger.info(f"Sync completed in {elapsed_time:.1f}s")

            except Exception as e:
                logger.error(f"Sync failed: {str(e)}")
                error_embed = discord.Embed(
                    title="❌ Sync Failed",
                    description=f"Error: {str(e)}",
                    color=discord.Color.red(),
                    timestamp=datetime.now()
                )
                await progress_message.edit(embed=error_embed)
            finally:
                try:
                    smb.close()
                except:
                    pass
        
        @self.tree.command(name="notifications", description="Toggle Discord notifications for motion events")
        async def toggle_notifications(interaction: discord.Interaction):
            self.notifications_enabled = not self.notifications_enabled
            status = "enabled" if self.notifications_enabled else "disabled"
            
            embed = discord.Embed(
                title="🔔 Notification Settings",
                description=f"Discord notifications have been {status}",
                color=discord.Color.green() if self.notifications_enabled else discord.Color.orange(),
                timestamp=datetime.now()
            )
            embed.add_field(
                name="Status", 
                value="🔔 Enabled" if self.notifications_enabled else "🔕 Disabled",
                inline=True
            )
            embed.add_field(
                name="Changed By",
                value=f"👤 {interaction.user.mention}",
                inline=True
            )
            embed.set_footer(text="Motion detection remains active")
            
            await interaction.response.send_message(embed=embed)
            logger.info(f"Discord notifications {status} by {interaction.user}")

        @self.tree.command(name="start_clock", description="Start time tracking")
        async def start_clock(
            interaction: discord.Interaction,
            start_time: str = None,
            auto_checkout_hours: int = 5
        ):
            if interaction.user.id != self.owner_id:
                await interaction.response.send_message("Only the owner can use this command.", ephemeral=True)
                return

            # Check if there's already an active session
            if self.active_clock or self.timesheet_data.get("active_session"):
                embed = discord.Embed(
                    title="⚠️ Clock Already Active",
                    description="You already have an active time tracking session",
                    color=discord.Color.yellow(),
                    timestamp=datetime.now()
                )
                if self.active_clock:
                    start = self.active_clock["start_time"]
                    duration = (datetime.now() - start).total_seconds() / 3600
                    embed.add_field(
                        name="Current Session",
                        value=f"Started: {start.strftime('%I:%M %p')}\nDuration: {round(duration, 2)} hours",
                        inline=False
                    )
                await interaction.response.send_message(embed=embed, ephemeral=True)
                return

            current_time = datetime.now()
            if start_time:
                try:
                    hour, minute = map(int, start_time.split(':'))
                    start_datetime = current_time.replace(hour=hour, minute=minute)
                except:
                    await interaction.response.send_message("Invalid time format. Use HH:MM", ephemeral=True)
                    return
            else:
                start_datetime = current_time

            auto_checkout_datetime = start_datetime + timedelta(hours=auto_checkout_hours)
            
            # Create active session
            self.active_clock = {
                "start_time": start_datetime,
                "auto_checkout_time": auto_checkout_datetime
            }
            
            # Store in timesheet data
            self.timesheet_data["active_session"] = {
                "start_time": start_datetime.strftime("%Y-%m-%d %H:%M:%S"),
                "auto_checkout_time": auto_checkout_datetime.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Save immediately
            self.save_timesheet()

            embed = discord.Embed(
                title="🕐 Clock Started",
                description="Time tracking has begun",
                color=discord.Color.green(),
                timestamp=datetime.now()
            )
            embed.add_field(name="Start Time", value=start_datetime.strftime("%I:%M %p"), inline=True)
            embed.add_field(name="Auto Checkout", value=f"In {auto_checkout_hours} hours", inline=True)
            
            await interaction.response.send_message(embed=embed)
            
            # Schedule auto-checkout
            await asyncio.sleep((auto_checkout_datetime - datetime.now()).total_seconds())
            if self.active_clock:  # If still active
                await self.auto_checkout(interaction.user)

        @self.tree.command(name="stop_clock", description="Stop time tracking")
        async def stop_clock(
            interaction: discord.Interaction,
            end_time: str = None
        ):
            if interaction.user.id != self.owner_id:
                await interaction.response.send_message("Only the owner can use this command.", ephemeral=True)
                return

            if not self.active_clock:
                await interaction.response.send_message("No active clock session found.")
                return

            current_time = datetime.now()
            if end_time:
                try:
                    hour, minute = map(int, end_time.split(':'))
                    end_datetime = current_time.replace(hour=hour, minute=minute)
                except:
                    await interaction.response.send_message("Invalid time format. Use HH:MM", ephemeral=True)
                    return
            else:
                end_datetime = current_time

            duration = (end_datetime - self.active_clock["start_time"]).total_seconds() / 3600

            # Add entry to timesheet
            entry = {
                "date": current_time.strftime("%Y-%m-%d"),
                "time_in": self.active_clock["start_time"].strftime("%H:%M"),
                "time_out": end_datetime.strftime("%H:%M"),
                "duration": round(duration, 2),
                "auto_checkout": False
            }

            self.timesheet_data["entries"].append(entry)
            self.timesheet_data["total_hours"] = round(
                sum(entry["duration"] for entry in self.timesheet_data["entries"]), 2
            )
            
            # Clear active session
            self.active_clock = None
            self.timesheet_data["active_session"] = None
            
            self.save_timesheet()

            embed = discord.Embed(
                title="⏱️ Clock Stopped",
                description="Time tracking has ended",
                color=discord.Color.orange(),
                timestamp=datetime.now()
            )
            embed.add_field(name="Duration", value=f"{round(duration, 2)} hours", inline=True)
            
            await interaction.response.send_message(embed=embed)

        @self.tree.command(name="timesheet_stats", description="View timesheet statistics for all or specific date")
        async def timesheet_stats(
            interaction: discord.Interaction,
            specific_date: str = None  # Optional date parameter in YYYY-MM-DD format
        ):
            if interaction.user.id != self.owner_id:
                await interaction.response.send_message("Only the owner can use this command.", ephemeral=True)
                return

            embed = discord.Embed(
                title="📊 Timesheet Statistics",
                color=discord.Color.blue(),
                timestamp=datetime.now()
            )

            # If specific date is provided, validate and show stats for that day
            if specific_date:
                try:
                    # Validate date format
                    datetime.strptime(specific_date, "%Y-%m-%d")
                    
                    # Get entries for specific date
                    day_entries = [
                        entry for entry in self.timesheet_data["entries"]
                        if entry["date"] == specific_date
                    ]
                    
                    if day_entries:
                        total_day_hours = sum(entry["duration"] for entry in day_entries)
                        
                        embed.add_field(
                            name=f"📅 {specific_date}",
                            value=f"Total Hours: {round(total_day_hours, 2)}",
                            inline=False
                        )
                        
                        # Add detailed entries
                        entries_text = ""
                        for entry in day_entries:
                            auto_checkout = "🤖" if entry["auto_checkout"] else "👤"
                            entries_text += f"{entry['time_in']} - {entry['time_out']} ({entry['duration']}h) {auto_checkout}\n"
                        
                        embed.add_field(
                            name="⏰ Time Entries",
                            value=f"```\n{entries_text}```",
                            inline=False
                        )
                    else:
                        embed.add_field(
                            name="❌ No Data",
                            value=f"No entries found for {specific_date}",
                            inline=False
                        )
                except:
                    embed.add_field(
                        name="❌ Invalid Date",
                        value="Date format must be YYYY-MM-DD",
                        inline=False
                    )
            else:
                # Calculate weekly hours
                current_week = datetime.now().isocalendar()[1]
                weekly_hours = sum(
                    entry["duration"] for entry in self.timesheet_data["entries"]
                    if datetime.strptime(entry["date"], "%Y-%m-%d").isocalendar()[1] == current_week
                )
                
                embed.add_field(
                    name="This Week",
                    value=f"{round(weekly_hours, 2)} hours",
                    inline=False
                )
                
                # Format total time
                total_hours = self.timesheet_data["total_hours"]
                days = total_hours // 24
                weeks = days // 7
                remaining_days = days % 7
                remaining_hours = total_hours % 24
                
                embed.add_field(
                    name="Total Time",
                    value=f"{int(weeks)} weeks, {int(remaining_days)} days, {round(remaining_hours, 2)} hours",
                    inline=False
                )

            # Show current session if active
            if self.active_clock:
                current_duration = (datetime.now() - self.active_clock["start_time"]).total_seconds() / 3600
                embed.add_field(
                    name="Current Session",
                    value=f"Running for {round(current_duration, 2)} hours",
                    inline=False
                )

            await interaction.response.send_message(embed=embed)

        @self.tree.command(name="imperial_march", description="Play the Imperial March on buzzer")
        async def play_imperial(interaction: discord.Interaction):
            if not BUZZER_ENABLED:
                await interaction.response.send_message("Buzzer is disabled")
                return
                
            await interaction.response.send_message("🎵 Playing Imperial March...")
            
            # Extended Imperial March sequence
            tones = [
                # First phrase
                (440, 0.5), (440, 0.5), (440, 0.5),          # A A A
                (349.2, 0.375), (523.3, 0.25),              # F C
                (440, 0.5), (349.2, 0.375), (523.3, 0.25),  # A F C
                (440, 1.0),                                   # A (half)

                # Second phrase (higher octave)
                (659.3, 0.5), (659.3, 0.5), (659.3, 0.5),    # E' E' E'
                (698.5, 0.375), (523.3, 0.25),              # F' c
                (415.3, 0.5), (349.2, 0.375), (523.3, 0.25),# Ab f c
                (440, 1.0),                                   # A (half)

                # Main theme
                (880, 0.5), (440, 0.375), (440, 0.25),      # A' A A
                (880, 0.5), (831.6, 0.375), (784, 0.25),    # A' Ab' G'
                (740, 0.125), (698.5, 0.125),                # Gb' F'
                (740, 0.25), (1,0.25), (466.2, 0.25),                # gb' Bb
                (622.3, 0.5), (587.3, 0.375), (554.4, 0.25),# Eb D Db

                # Final section
                (523.3, 0.125), (493.9, 0.125),              # c B
                (523.3, 0.25), (1,0.25), (349.2, 0.25),              # c f
                (415.3, 0.5), (349.2, 0.25),                # Ab f
                (415.3, 0.25), (523.3, 0.5),                # Ab C
                (440, 0.375), (523.3, 0.25),                # a C
                (659.3, 2.0)                                  # e (half)
            ]
            
            try:
                for freq, duration in tones:
                    self.play_buzzer_tone(freq, duration)
                    sleep(0.05)  # Small pause between notes
                    
                GPIO.cleanup(BUZZER_PIN)
            except Exception as e:
                logger.error(f"Buzzer playback error: {e}")
                await interaction.followup.send("❌ Error playing buzzer sequence")

        @self.tree.command(name="generate_heatmap", description="Generate motion heatmap")
        async def generate_heatmap(
            interaction: discord.Interaction,
            hours: int = 24,
            alpha: float = 0.6
        ):
            """Generate and send motion heatmap"""
            await interaction.response.defer()

            try:
                # Create visualization
                plt.figure(figsize=(12, 8))
                plt.imshow(self.heatmap, cmap='hot', interpolation='gaussian')
                plt.colorbar(label='Motion Intensity')
                
                # Add timestamp and stats
                total_motion = np.sum(self.heatmap > 0)
                coverage = (total_motion / self.heatmap.size) * 100
                
                plt.title(f"Motion Heatmap (Last {hours} Hours)\n"
                        f"Coverage: {coverage:.1f}% of frame")
                
                # Save plot
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                heatmap_path = f"heatmap_{timestamp}.png"
                plt.savefig(heatmap_path)
                plt.close()

                # Create embed
                embed = discord.Embed(
                    title="🔥 Motion Heatmap Analysis",
                    description="Visualization of motion patterns",
                    color=discord.Color.orange(),
                    timestamp=datetime.now()
                )
                
                embed.add_field(
                    name="📊 Statistics",
                    value=f"""```
    Time Range: {hours} hours
    Motion Coverage: {coverage:.1f}%
    Peak Activity: {np.max(self.heatmap):.0f} events
    Total Events: {np.sum(self.heatmap):.0f}
    ```""",
                    inline=False
                )

                # Send to Discord
                file = discord.File(heatmap_path, filename="heatmap.png")
                embed.set_image(url="attachment://heatmap.png")
                
                await interaction.followup.send(embed=embed, file=file)
                
                # Cleanup
                os.remove(heatmap_path)
                logger.info("Heatmap generated and sent successfully")

            except Exception as e:
                logger.error(f"Error generating heatmap: {e}")
                error_embed = discord.Embed(
                    title="❌ Heatmap Generation Failed",
                    description=f"Error: {str(e)}",
                    color=discord.Color.red()
                )
                await interaction.followup.send(embed=error_embed)

        @self.tree.command(name="train_heatmap", description="Analyze motion patterns and optimize detection")
        async def train_system(
            interaction: discord.Interaction,
            days: int = 7,
            threshold_adjust: bool = True,
            intensity_scale: float = 0.1  # Add intensity scaling parameter
        ):
            """Analyze captured images to optimize detection settings"""
            await interaction.response.defer()
            
            start_time = time_module.time()
            
            # Initial embed
            embed = discord.Embed(
                title="🧠 Training System",
                description="Analyzing motion patterns...",
                color=discord.Color.blue(),
                timestamp=datetime.now()
            )
            progress_msg = await interaction.followup.send(embed=embed)
            
            try:
                # Get date range
                end_date = datetime.now()
                start_date = end_date - timedelta(days=days)
                
                # Collect image paths
                image_paths = []
                motion_data = []
                
                for date in (start_date + timedelta(n) for n in range(days)):
                    folder = f"captures_{date.strftime('%Y%m%d')}"
                    if os.path.exists(folder):
                        paths = sorted(glob.glob(f"{folder}/*.jpg"))
                        image_paths.extend(paths)
                        
                        # Update progress
                        embed.description = f"Found {len(image_paths)} images to analyze..."
                        await progress_msg.edit(embed=embed)
                
                if not image_paths:
                    embed.title = "❌ Analysis Failed"
                    embed.description = "No images found in the specified date range"
                    embed.color = discord.Color.red()
                    await progress_msg.edit(embed=embed)
                    return
                
                # Initialize motion analysis arrays
                total_motion_mask = np.zeros((480, 640), dtype=np.float32)  # Use float32 for better precision
                time_distribution = []
                areas = []
                
                for i, path in enumerate(image_paths):
                    if i % 10 == 0:  # Update progress every 10 images
                        embed.description = f"Processing image {i+1}/{len(image_paths)}..."
                        await progress_msg.edit(embed=embed)
                        
                    # Extract timestamp from filename
                    timestamp = datetime.strptime(os.path.basename(path)[:15], "%Y%m%d_%H%M%S")
                    time_distribution.append(timestamp.hour)
                    
                    # Process image
                    frame = cv2.imread(path)
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray = cv2.GaussianBlur(gray, (21, 21), 0)
                    
                    # Detect motion areas
                    thresh = cv2.threshold(gray, self.motion_threshold, 255, cv2.THRESH_BINARY)[1]
                    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        area = cv2.contourArea(contour)
                        if area > self.min_area:
                            areas.append(area)
                            mask = np.zeros_like(gray, dtype=np.float32)
                            cv2.drawContours(mask, [contour], -1, intensity_scale, -1)  # Use intensity_scale for accumulation
                            total_motion_mask += mask
                            
                            # Store motion data for clustering
                            x, y, w, h = cv2.boundingRect(contour)
                            center_x = x + w//2
                            center_y = y + h//2
                            motion_data.append([center_x, center_y, timestamp.hour])
                
                # Normalize motion mask
                if np.max(total_motion_mask) > 0:
                    total_motion_mask = (total_motion_mask / np.max(total_motion_mask)) * 255
                
                # Cluster analysis
                if motion_data:
                    X = np.array(motion_data)
                    db = DBSCAN(eps=30, min_samples=5).fit(X[:, :2])  # Cluster spatial data
                    n_clusters = len(set(db.labels_)) - (1 if -1 in db.labels_ else 0)
                    
                    # Calculate optimal threshold
                    if threshold_adjust and areas:
                        suggested_threshold = np.percentile(areas, 25)  # Use 25th percentile
                        suggested_min_area = max(100, int(suggested_threshold))
                    
                    # Generate visualization
                    plt.figure(figsize=(15, 10))
                    
                    # Motion heatmap with improved visualization
                    plt.subplot(2, 2, 1)
                    sns.heatmap(
                        total_motion_mask, 
                        cmap='hot',
                        vmin=0,
                        vmax=255,
                        robust=True,  # Handle outliers
                        cbar_kws={'label': 'Motion Intensity (normalized)'}
                    )
                    plt.title('Motion Heatmap')
                    
                    # Time distribution
                    plt.subplot(2, 2, 2)
                    sns.histplot(time_distribution, bins=24)
                    plt.title('Activity by Hour')
                    plt.xlabel('Hour of Day')
                    
                    # Motion area distribution
                    plt.subplot(2, 2, 3)
                    sns.histplot(areas, bins=50)
                    plt.title('Motion Area Distribution')
                    plt.xlabel('Area (pixels)')
                    
                    # Cluster visualization
                    plt.subplot(2, 2, 4)
                    scatter = plt.scatter(X[:, 0], X[:, 1], c=db.labels_, cmap='viridis')
                    plt.title(f'Motion Clusters (n={n_clusters})')
                    plt.colorbar(scatter)
                    
                    # Save and send plot
                    analysis_path = "motion_analysis.png"
                    plt.tight_layout()
                    plt.savefig(analysis_path)
                    plt.close()
                    
                    # Create results embed
                    results_embed = discord.Embed(
                        title="📊 Motion Analysis Results",
                        description="System analysis complete",
                        color=discord.Color.green(),
                        timestamp=datetime.now()
                    )
                    
                    # Add statistics
                    peak_hour = max(Counter(time_distribution).items(), key=lambda x: x[1])[0]
                    avg_area = np.mean(areas)
                    
                    results_embed.add_field(
                        name="📈 Activity Patterns",
                        value=f"""```
        Images Analyzed: {len(image_paths)}
        Peak Activity: {peak_hour:02d}:00
        Motion Clusters: {n_clusters}
        Avg Motion Area: {avg_area:.0f}px²
        ```""",
                        inline=False
                    )
                    
                    if threshold_adjust:
                        results_embed.add_field(
                            name="⚙️ Recommended Settings",
                            value=f"""```
        Min Area: {suggested_min_area} (current: {self.min_area})
        ```""",
                            inline=False
                        )
                    
                    # Calculate reliability score
                    false_positives = sum(1 for area in areas if area < self.min_area)
                    reliability = 1 - (false_positives / len(areas) if areas else 0)
                    
                    results_embed.add_field(
                        name="🎯 System Reliability",
                        value=f"""```
        Score: {reliability*100:.1f}%
        False Positives: {false_positives}
        Motion Intensity Scale: {intensity_scale}
        ```""",
                        inline=False
                    )
                    
                    # Add processing time
                    elapsed = time_module.time() - start_time
                    results_embed.set_footer(text=f"Analysis completed in {elapsed:.1f}s")
                    
                    # Send results
                    file = discord.File(analysis_path, filename="analysis.png")
                    results_embed.set_image(url="attachment://analysis.png")
                    
                    await progress_msg.edit(embed=results_embed)
                    await interaction.followup.send(file=file)
                    
                    # Cleanup
                    os.remove(analysis_path)
                    
                    # Store analysis results
                    self.last_analysis = {
                        'timestamp': datetime.now(),
                        'reliability': reliability,
                        'suggested_min_area': suggested_min_area if threshold_adjust else None,
                        'peak_hour': peak_hour,
                        'clusters': n_clusters,
                        'intensity_scale': intensity_scale
                    }
                    
                else:
                    embed.title = "❌ Analysis Failed"
                    embed.description = "No motion data found in images"
                    embed.color = discord.Color.red()
                    await progress_msg.edit(embed=embed)
                    
            except Exception as e:
                logger.error(f"Training analysis failed: {e}")
                embed.title = "❌ Analysis Failed"
                embed.description = f"Error: {str(e)}"
                embed.color = discord.Color.red()
                await progress_msg.edit(embed=embed)
        
        @self.tree.command(name="send_mail", description="Send timesheet report via email")
        async def send_mail(interaction: discord.Interaction):
            """Send timesheet statistics via email"""
            if interaction.user.id != self.owner_id:
                await interaction.response.send_message("Only the owner can use this command.", ephemeral=True)
                return
                
            await interaction.response.defer()
            
            try:
                import smtplib
                from email.mime.text import MIMEText
                from email.mime.multipart import MIMEMultipart
                
                # Get email settings from env
                sender_email = os.getenv('EMAIL_ADDRESS')
                receiver_email = os.getenv('Email_RECEIVER')  # Sending to whoever you want
                password = os.getenv('EMAIL_PASSWORD')
                smtp_server = os.getenv('SMTP_SERVER')
                smtp_port = int(os.getenv('SMTP_PORT', 587))
                
                if not all([sender_email, password, smtp_server]):
                    raise ValueError("Missing email configuration in cred.env")
                    
                # Calculate statistics
                total_hours = self.timesheet_data["total_hours"]
                raw_hours = sum(entry["duration"] for entry in self.timesheet_data["entries"])
                
                # Calculate weekly hours
                current_week = datetime.now().isocalendar()[1]
                weekly_hours = sum(
                    entry["duration"] for entry in self.timesheet_data["entries"]
                    if datetime.strptime(entry["date"], "%Y-%m-%d").isocalendar()[1] == current_week
                )
                
                # Format readable duration
                def format_hours(hours):
                    weeks = int(hours // (24 * 7))
                    remaining = hours % (24 * 7)
                    days = int(remaining // 24)
                    remaining_hours = remaining % 24
                    
                    parts = []
                    if weeks: parts.append(f"{weeks} weeks")
                    if days: parts.append(f"{days} days")
                    if remaining_hours: parts.append(f"{remaining_hours:.1f} hours")
                    return ", ".join(parts)
                
                # Create message
                msg = MIMEMultipart()
                msg['From'] = sender_email
                msg['To'] = receiver_email
                msg['Subject'] = f"Timesheet Report - {datetime.now().strftime('%Y-%m-%d')}"
                
                body = f"""
                <h2>🕒 Timesheet Statistics</h2>
                
                <h3>Current Week</h3>
                Raw Hours: {weekly_hours:.2f} hours
                Formatted: {format_hours(weekly_hours)}
                
                <h3>All Time</h3>
                Raw Hours: {raw_hours:.2f} hours
                Cleaned Total: {total_hours:.2f} hours
                Formatted: {format_hours(total_hours)}
                
                <h3>Active Session</h3>
                {
                    f"Running since: {self.active_clock['start_time'].strftime('%I:%M %p')}" 
                    if self.active_clock else "No active session"
                }
                
                <br><br>
                <small>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</small>
                """
                
                msg.attach(MIMEText(body, 'html'))
                
                # Send email
                with smtplib.SMTP(smtp_server, smtp_port) as server:
                    server.starttls()
                    server.login(sender_email, password)
                    server.send_message(msg)
                    
                # Success message
                embed = discord.Embed(
                    title="📧 Email Sent",
                    description="Timesheet report has been emailed successfully",
                    color=discord.Color.green(),
                    timestamp=datetime.now()
                )
                embed.add_field(
                    name="Statistics Sent",
                    value=f"""```
        Weekly Hours: {weekly_hours:.2f}
        Total Hours: {total_hours:.2f}
        Raw Hours: {raw_hours:.2f}
        ```""",
                    inline=False
                )
                embed.add_field(
                    name="Recipient",
                    value=f"`{receiver_email}`",
                    inline=False
                )
                
                await interaction.followup.send(embed=embed)
                logger.info(f"Timesheet report sent to {receiver_email}")
                
            except Exception as e:
                error_embed = discord.Embed(
                    title="❌ Email Failed",
                    description=f"Failed to send email: {str(e)}",
                    color=discord.Color.red()
                )
                await interaction.followup.send(embed=error_embed)
                logger.error(f"Failed to send email: {str(e)}")
        @self.tree.command(name="list_entries", description="List recent timesheet entries")
        async def list_entries(interaction: discord.Interaction, count: int = 5):
            """List the most recent timesheet entries"""
            if interaction.user.id != self.owner_id:
                await interaction.response.send_message("Only the owner can use this command.", ephemeral=True)
                return

            entries = self.timesheet_data["entries"][-count:]
            
            embed = discord.Embed(
                title="📋 Recent Timesheet Entries",
                color=discord.Color.blue(),
                timestamp=datetime.now()
            )
            
            for i, entry in enumerate(entries):
                embed.add_field(
                    name=f"Entry #{len(self.timesheet_data['entries']) - count + i + 1}",
                    value=f"""```
        Date: {entry['date']}
        Time: {entry['time_in']} - {entry['time_out']}
        Duration: {entry['duration']}h
        Auto: {'Yes' if entry.get('auto_checkout', False) else 'No'}```""",
                    inline=False
                )
            
            await interaction.response.send_message(embed=embed)

        @self.tree.command(name="remove_entry", description="Remove a timesheet entry")
        async def remove_entry(interaction: discord.Interaction, entry_number: int):
            """Remove a specific entry from the timesheet"""
            if interaction.user.id != self.owner_id:
                await interaction.response.send_message("Only the owner can use this command.", ephemeral=True)
                return

            try:
                if entry_number < 1 or entry_number > len(self.timesheet_data["entries"]):
                    await interaction.response.send_message("Invalid entry number.", ephemeral=True)
                    return

                # Get the entry to remove
                entry = self.timesheet_data["entries"][entry_number - 1]
                
                # Remove the entry
                self.timesheet_data["entries"].pop(entry_number - 1)
                
                # Recalculate total hours
                self.timesheet_data["total_hours"] = round(
                    sum(entry["duration"] for entry in self.timesheet_data["entries"]), 2
                )
                
                # Save changes
                self.save_timesheet()
                
                embed = discord.Embed(
                    title="✅ Entry Removed",
                    description=f"Entry #{entry_number} has been removed",
                    color=discord.Color.green(),
                    timestamp=datetime.now()
                )
                embed.add_field(
                    name="Removed Entry",
                    value=f"""```
        Date: {entry['date']}
        Time: {entry['time_in']} - {entry['time_out']}
        Duration: {entry['duration']}h```""",
                    inline=False
                )
                
                await interaction.response.send_message(embed=embed)
                
            except Exception as e:
                await interaction.response.send_message(f"Error removing entry: {str(e)}", ephemeral=True)

        @self.tree.command(name="edit_entry", description="Edit a timesheet entry")
        async def edit_entry(
            interaction: discord.Interaction, 
            entry_number: int,
            date: str = None,
            time_in: str = None,
            time_out: str = None
        ):
            """Edit an existing timesheet entry"""
            if interaction.user.id != self.owner_id:
                await interaction.response.send_message("Only the owner can use this command.", ephemeral=True)
                return

            try:
                if entry_number < 1 or entry_number > len(self.timesheet_data["entries"]):
                    await interaction.response.send_message("Invalid entry number.", ephemeral=True)
                    return

                # Get the entry to edit
                entry = self.timesheet_data["entries"][entry_number - 1]
                old_entry = entry.copy()
                
                # Update fields if provided
                if date:
                    try:
                        datetime.strptime(date, "%Y-%m-%d")
                        entry["date"] = date
                    except ValueError:
                        await interaction.response.send_message("Invalid date format. Use YYYY-MM-DD", ephemeral=True)
                        return
                        
                if time_in:
                    try:
                        datetime.strptime(time_in, "%H:%M")
                        entry["time_in"] = time_in
                    except ValueError:
                        await interaction.response.send_message("Invalid time_in format. Use HH:MM", ephemeral=True)
                        return
                        
                if time_out:
                    try:
                        datetime.strptime(time_out, "%H:%M")
                        entry["time_out"] = time_out
                    except ValueError:
                        await interaction.response.send_message("Invalid time_out format. Use HH:MM", ephemeral=True)
                        return

                # Recalculate duration
                if time_in or time_out:
                    time_in_dt = datetime.strptime(entry["time_in"], "%H:%M")
                    time_out_dt = datetime.strptime(entry["time_out"], "%H:%M")
                    if time_out_dt < time_in_dt:  # Handle overnight shifts
                        time_out_dt += timedelta(days=1)
                    duration = (time_out_dt - time_in_dt).total_seconds() / 3600
                    entry["duration"] = round(duration, 2)

                # Recalculate total hours
                self.timesheet_data["total_hours"] = round(
                    sum(entry["duration"] for entry in self.timesheet_data["entries"]), 2
                )
                
                # Save changes
                self.save_timesheet()
                
                embed = discord.Embed(
                    title="✅ Entry Updated",
                    description=f"Entry #{entry_number} has been updated",
                    color=discord.Color.green(),
                    timestamp=datetime.now()
                )
                embed.add_field(
                    name="Old Entry",
                    value=f"""```
        Date: {old_entry['date']}
        Time: {old_entry['time_in']} - {old_entry['time_out']}
        Duration: {old_entry['duration']}h```""",
                    inline=True
                )
                embed.add_field(
                    name="New Entry",
                    value=f"""```
        Date: {entry['date']}
        Time: {entry['time_in']} - {entry['time_out']}
        Duration: {entry['duration']}h```""",
                    inline=True
                )
                
                await interaction.response.send_message(embed=embed)
                
            except Exception as e:
                await interaction.response.send_message(f"Error editing entry: {str(e)}", ephemeral=True)

    def format_uptime(self):
        """Format the system uptime"""
        uptime = time_module.time() - self.start_time
        hours, remainder = divmod(int(uptime), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours}h {minutes}m {seconds}s"

    def format_last_motion_time(self):
        """Format the last motion time for status display"""
        if self.last_motion_time == 0:
            return "No motion detected yet"
        time_diff = time_module.time() - self.last_motion_time
        return f"{int(time_diff)} seconds ago"

    async def send_snapshot(self, interaction):
        """Take and send a snapshot directly to Discord"""
        logger.info("Taking snapshot")
        start_time = time_module.time()
        ret, frame = self.camera.read()
        if not ret:
            logger.error("Failed to capture snapshot")
            await interaction.response.send_message("Failed to capture snapshot")
            return

        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            logger.error("Failed to encode snapshot")
            await interaction.response.send_message("Failed to process image")
            return

        io_buf = io.BytesIO(buffer)
        discord_file = discord.File(fp=io_buf, filename='snapshot.jpg')
        
        await interaction.response.send_message(
            "Current snapshot:",
            file=discord_file
        )
        elapsed_time = time_module.time() - start_time
        logger.info(f"Snapshot captured and sent in {elapsed_time:.2f}s")

    def detect_motion(self, frame):
        """Detect motion in frame with cooldown period and return contours"""
        current_time = time_module.time()
        if current_time - self.last_motion_time < self.motion_cooldown:
            return False, []

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.background is None:
            self.background = gray
            logger.info("Initial background frame captured")
            return False, []

        frame_delta = cv2.absdiff(self.background, gray)
        thresh = cv2.threshold(frame_delta, self.motion_threshold, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        motion_detected = False
        significant_contours = []
        
        for contour in contours:
            if cv2.contourArea(contour) > self.min_area:
                self.last_motion_time = current_time
                motion_detected = True
                significant_contours.append(contour)
                logger.info(f"Motion detected! Area: {cv2.contourArea(contour):.2f}")
                asyncio.create_task(self.play_motion_alert())
                current_time = time_module.time()
                if not hasattr(self, '_last_heatmap_update') or \
                current_time - self._last_heatmap_update > self.heatmap_cooldown:
                    asyncio.create_task(self.update_heatmap(significant_contours))
                    self._last_heatmap_update = current_time
                if not self.notifications_enabled and \
                (not hasattr(self, '_last_alert_time') or \
                    current_time - self._last_alert_time > 60):  # 30 minutes = 1800 seconds
                    self._last_alert_time = current_time
        

        self.background = gray
        return motion_detected, significant_contours

    async def upload_to_smb(self, file_path, max_retries=3):
        """Upload file to SMB server with retry logic"""
        for attempt in range(max_retries):
            try:
                # Try to connect
                if not self.smb_client.connect(
                    os.getenv('SMB_SERVER_IP'),
                    int(os.getenv('SMB_PORT', 445))
                ):
                    raise Exception("Failed to connect to SMB server")

                smb_path = os.path.basename(file_path)
                share_name = os.getenv('SMB_SHARE_NAME')
                
                # Create directory if needed
                try:
                    self.smb_client.createDirectory(share_name, "timelapse")
                except:
                    pass
                    
                with open(file_path, 'rb') as file:
                    success = self.smb_client.storeFile(
                        share_name,
                        f"timelapse/{smb_path}",
                        file
                    )
                    if not success:
                        raise Exception("Failed to store file on SMB share")
                        
                logger.info(f"Successfully uploaded {file_path} to SMB server")
                return True
                
            except Exception as e:
                logger.error(f"SMB upload attempt {attempt + 1} failed: {str(e)}")
                # Create new connection for retry
                self.smb_client = SMBConnection(
                    os.getenv('SMB_USERNAME'),
                    os.getenv('SMB_PASSWORD'),
                    os.getenv('CLIENT_NAME'),
                    os.getenv('SERVER_NAME'),
                    use_ntlm_v2=True
                )
                if attempt == max_retries - 1:
                    logger.error(f"All SMB upload attempts failed for {file_path}")
                    raise
                await asyncio.sleep(1)  # Wait before retry
            finally:
                try:
                    self.smb_client.close()
                except:
                    pass

    async def capture_and_save(self, frame):
        """Save motion-triggered frame and send to Discord with motion boxes"""
        start_time = time_module.time()
        current_time = time_module.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"captures_{datetime.now().strftime('%Y%m%d')}"
        os.makedirs(folder_name, exist_ok=True)
        image_path = f"{folder_name}/{timestamp}.jpg"
        
        if not self.is_fully_initialized:
            logger.info("Skipping capture - system not fully initialized")
            return
        
        if current_time - self.last_capture_time < self.capture_interval:
            logger.info(f"Skipping capture - waiting for {self.capture_interval - (current_time - self.last_capture_time):.1f}s")
            return
        
        self.last_capture_time = current_time
        start_time = current_time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            # Save original frame without annotations first
            cv2.imwrite(image_path, frame)
            logger.info(f"Saved original capture to {image_path}")
            
            # Upload original to SMB
            await self.upload_to_smb(image_path)
            
            # Only send Discord notification if enabled
            if self.notifications_enabled:
                # Create annotated version only for Discord
                motion_detected, contours = self.detect_motion(frame)
                discord_frame = frame.copy()
                
                # Draw motion boxes only on Discord version
                for contour in contours:
                    (x, y, w, h) = cv2.boundingRect(contour)
                    cv2.rectangle(
                        discord_frame,
                        (x, y),
                        (x + w, y + h),
                        (0, 0, 255),  # BGR color (Red)
                        2  # Line thickness
                    )
                
                # Save temporary annotated version for Discord
                discord_image_path = f"{folder_name}/temp_{timestamp}.jpg"
                cv2.imwrite(discord_image_path, discord_frame)
                
                # Send to Discord
                channel = self.discord_client.get_channel(int(os.getenv('DISCORD_CHANNEL_ID')))
                if not channel:
                    raise Exception("Discord channel not found")
                        
                embed = discord.Embed(
                    title="🚨 Motion Detected!",
                    description="New movement captured by the camera system",
                    color=discord.Color.red(),
                    timestamp=datetime.now()
                )
                
                # Add fields with detection details
                embed.add_field(
                    name="📅 Timestamp",
                    value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    inline=True
                )
                embed.add_field(
                    name="📁 File",
                    value=f"`{timestamp}.jpg`",
                    inline=True
                )
                embed.add_field(
                    name="⏱️ Response Time",
                    value=f"{(time_module.time() - self.last_motion_time):.2f}s",
                    inline=True
                )
                
                embed.add_field(
                    name="📊 Motion Areas",
                    value=f"{len(contours)} detected",
                    inline=True
                )
                
                embed.add_field(
                    name="🎥 Camera Status",
                    value="Active" if self.is_monitoring else "Paused",
                    inline=False
                )
                
                file = discord.File(discord_image_path, filename="motion.jpg")
                embed.set_image(url="attachment://motion.jpg")
                
                await channel.send(embed=embed, file=file)
                
                # Clean up temporary Discord image
                os.remove(discord_image_path)
            
            elapsed_time = time_module.time() - start_time
            logger.info(f"Motion capture processed in {elapsed_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error in capture_and_save: {str(e)}", exc_info=True)

    async def create_timelapse_video(self, fps=30, date=None):
        """Create video from captured images"""
        try:
            folder_name = f"captures_{date}" if date else f"captures_{datetime.now().strftime('%Y%m%d')}"
            
            if not os.path.exists(folder_name):
                logger.error(f"Folder {folder_name} not found")
                return None

            images = sorted([img for img in os.listdir(folder_name) if img.endswith(".jpg")])
            if not images:
                logger.error(f"No images found in {folder_name}")
                return None

            first_image = cv2.imread(os.path.join(folder_name, images[0]))
            height, width, _ = first_image.shape  # Changed to use _ for unused layers variable
            video_path = f"{folder_name}_timelapse.mp4"
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

            for image in images:
                video.write(cv2.imread(os.path.join(folder_name, image)))

            video.release()
            logger.info(f"Created timelapse video: {video_path}")
            return video_path

        except Exception as e:
            logger.error(f"Failed to create timelapse video: {str(e)}")
            return None

    async def monitor(self):
        """Main monitoring loop"""
        logger.info("Starting motion monitoring")
        retry_count = 0
        max_retries = 3
        
        while not self.is_fully_initialized:
            await asyncio.sleep(1)
            continue
            
        while True:
            if not self.is_monitoring:
                await asyncio.sleep(1)
                continue
                
            try:
                ret, frame = self.camera.read()
                if not ret:
                    retry_count += 1
                    logger.error(f"Failed to grab frame (attempt {retry_count}/{max_retries})")
                    if retry_count >= max_retries:
                        logger.critical("Camera appears to be disconnected")
                        self.camera.release()
                        await asyncio.sleep(2)
                        self.camera = cv2.VideoCapture(0)
                        retry_count = 0
                    await asyncio.sleep(1)
                    continue

                retry_count = 0
                
                has_motion, detected_contours = self.detect_motion(frame)
                if has_motion and detected_contours:  # Only process if we have both
                    await self.capture_and_save(frame)
                
            except Exception as e:
                logger.error(f"Error in monitor loop: {str(e)}")
                await asyncio.sleep(1)
            
            await asyncio.sleep(0.1)

    async def scheduled_capture(self):
        """Take scheduled snapshots every 10 minutes"""
        logger.info("Starting scheduled capture task")
        
        while True:
            if self.is_fully_initialized:
                try:
                    ret, frame = self.camera.read()
                    if ret:
                        # Save frame
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        folder_name = f"timelapse_photos_primary"
                        os.makedirs(folder_name, exist_ok=True)
                        image_path = f"{folder_name}/{timestamp}.jpg"
                        
                        # Save the frame
                        cv2.imwrite(image_path, frame)
                        logger.info(f"Scheduled capture saved to {image_path}")
                        
                        # Upload to SMB without Discord notification
                        try:
                            await self.upload_to_smb(image_path)
                        except Exception as e:
                            logger.error(f"Failed to upload scheduled capture: {str(e)}")
                            
                except Exception as e:
                    logger.error(f"Error in scheduled capture: {str(e)}")
                    
            # Wait for 30 minutes
            await asyncio.sleep(600)  # 30 minutes = 1800 seconds

    async def run_camera(self):
        """Run the camera monitoring"""
        try:
            await self.monitor()
        except Exception as e:
            logger.error(f"Error in camera monitoring: {str(e)}")
    
    async def auto_checkout(self, user):
        """Handle automatic checkout after specified duration"""
        if self.active_clock:
            try:
                duration = (datetime.now() - self.active_clock["start_time"]).total_seconds() / 3600
                
                # Add entry to timesheet
                entry = {
                    "date": datetime.now().strftime("%Y-%m-%d"),
                    "time_in": self.active_clock["start_time"].strftime("%H:%M"),
                    "time_out": datetime.now().strftime("%H:%M"),
                    "duration": round(duration, 2),
                    "auto_checkout": True
                }
                
                self.timesheet_data["entries"].append(entry)
                self.timesheet_data["total_hours"] = round(
                    sum(entry["duration"] for entry in self.timesheet_data["entries"]), 2
                )
                
                # Clear active session data
                self.active_clock = None
                self.timesheet_data["active_session"] = None
                
                # Save changes
                self.save_timesheet()

                # Send DM to owner
                try:
                    owner = await self.discord_client.fetch_user(self.owner_id)
                    if owner:
                        embed = discord.Embed(
                            title="⚠️ Auto Clock-Out",
                            description="You have been automatically clocked out",
                            color=discord.Color.yellow(),
                            timestamp=datetime.now()
                        )
                        embed.add_field(
                            name="Duration",
                            value=f"{round(duration, 2)} hours",
                            inline=True
                        )
                        await owner.send(embed=embed)
                except Exception as e:
                    logger.error(f"Failed to send auto-checkout notification: {str(e)}")
                    
            except Exception as e:
                logger.error(f"Error during auto-checkout: {str(e)}")
                # Force clear session data even if error occurs
                self.active_clock = None
                self.timesheet_data["active_session"] = None
                self.save_timesheet()

    async def play_motion_alert(self):
        """Play Imperial March with 10-minute cooldown"""
        if not BUZZER_ENABLED:
            return

        current_time = time_module.time()
        
        # Check if enough time has passed since last alert
        if hasattr(self, '_last_imperial_march_time') and \
        current_time - self._last_imperial_march_time < 3600:  # 3600 seconds = 1 hour
            logger.info("Skipping Imperial March alert - cooldown period")
            return

        try:
            # Update last play time
            self._last_imperial_march_time = current_time
            
            # Imperial March first phrase (simplified)
            notes = [
                # First phrase
                (440, 0.5), (440, 0.5), (440, 0.5),          # A A A
                (349.2, 0.375), (523.3, 0.25),              # F C
                (440, 0.5), (349.2, 0.375), (523.3, 0.25),  # A F C
                (440, 1.0),                                   # A (half)

                # Second phrase (higher octave)
                (659.3, 0.5), (659.3, 0.5), (659.3, 0.5),    # E' E' E'
                (698.5, 0.375), (523.3, 0.25),              # F' c
                (415.3, 0.5), (349.2, 0.375), (523.3, 0.25),# Ab f c
                (440, 1.0),                                   # A (half)

                # Main theme
                (880, 0.5), (440, 0.375), (440, 0.25),      # A' A A
                (880, 0.5), (831.6, 0.375), (784, 0.25),    # A' Ab' G'
                (740, 0.125), (698.5, 0.125),                # Gb' F'
                (740, 0.25), (1,0.25), (466.2, 0.25),                # gb' Bb
                (622.3, 0.5), (587.3, 0.375), (554.4, 0.25),# Eb D Db

                # Final section
                (523.3, 0.125), (493.9, 0.125),              # c B
                (523.3, 0.25), (1,0.25), (349.2, 0.25),              # c f
                (415.3, 0.5), (349.2, 0.25),                # Ab f
                (415.3, 0.25), (523.3, 0.5),                # Ab C
                (440, 0.375), (523.3, 0.25),                # a C
                (659.3, 2.0)                                  # e (half)
            ]
            
            for freq, duration in notes:
                self.play_buzzer_tone(freq, duration)
                await asyncio.sleep(0.02)  # Small gap between notes
                
        except Exception as e:
            logger.error(f"Error playing motion alert: {e}")
        finally:
            GPIO.cleanup(BUZZER_PIN)
                    
    def run(self):
        """Run the system"""
        try:
            self.start_time = time_module.time()
            loop = asyncio.get_event_loop()
            
            # Create initialization task
            init_task = loop.create_task(self.initialize())
            
            # Create other tasks
            loop.create_task(self.run_camera())
            # loop.create_task(self.scheduled_capture())   #add this line to enable scheduled capture
            
            # Start Discord client
            loop.run_until_complete(self.discord_client.start(os.getenv('DISCORD_TOKEN')))
            
        except KeyboardInterrupt:
            self.cleanup()
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            self.cleanup()

    async def shutdown_sequence(self):
        """Execute graceful shutdown sequence"""
        try:
            # Create shutdown stats
            uptime = time_module.time() - self.start_time
            hours, remainder = divmod(int(uptime), 3600)
            minutes, seconds = divmod(remainder, 60)
            
            # Count total captures
            total_captures = sum(len(files) for _, _, files in os.walk('captures_'))
            timelapse_photos = sum(len(files) for _, _, files in os.walk('timelapse_photos_primary'))
            
            goodbye_embed = discord.Embed(
                title="🔌 System Shutting Down",
                description="The timelapse monitoring system is going offline...",
                color=discord.Color.red(),
                timestamp=datetime.now()
            )
            
            # Add stats and status fields
            goodbye_embed.add_field(
                name="📊 Session Statistics",
                value=f"""```
    Uptime: {hours}h {minutes}m {seconds}s
    Total Captures: {total_captures:,}
    Timelapse Photos: {timelapse_photos:,}
    Last Motion: {self.format_last_motion_time()}```""",
                inline=False
            )
            
            goodbye_embed.add_field(
                name="💾 System Status",
                value=f"""```
    Camera: {'Connected' if self.camera.isOpened() else 'Disconnected'}
    Monitoring: {'Active' if self.is_monitoring else 'Paused'}
    Notifications: {'Enabled' if self.notifications_enabled else 'Disabled'}```""",
                inline=False
            )
            
            goodbye_embed.add_field(
                name="👋 Farewell",
                value="All files have been synced and resources cleaned up. Goodbye!",
                inline=False
            )
            
            goodbye_embed.set_footer(text=f"Shutdown initiated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

            # Send goodbye message if channel exists
            if self.discord_channel:
                await self.discord_channel.send(embed=goodbye_embed)
                logger.info("Goodbye message sent successfully")
            
            # Cleanup sequence
            logger.info("Beginning cleanup sequence...")
            
            if self.camera.isOpened():
                self.camera.release()
                logger.info("Camera released")
            
            cv2.destroyAllWindows()
            logger.info("OpenCV windows closed")
            
            await self.discord_client.close()
            logger.info("Discord connection closed")
            
            logger.info("Cleanup complete, shutting down...")
            
            # Exit process
            os._exit(0)
            
        except Exception as e:
            logger.error(f"Error during shutdown sequence: {str(e)}")
            os._exit(1)

    def cleanup(self):
        """Cleanup resources with fallback"""
        logger.info("Starting cleanup")
        try:

            #start shutdown task
            loop = asyncio.get_event_loop()
            loop.run_until_complete(self.shutdown_sequence())            
            # Fallback cleanup if Discord is not ready or shutdown sequence fails
            if hasattr(self, 'camera') and self.camera.isOpened():
                self.camera.release()
                logger.info("Camera released")
            cv2.destroyAllWindows()
            logger.info("OpenCV windows closed")
                
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
        finally:
            logger.info("System shut down complete")

if __name__ == "__main__":
    try:
        logger.info("Starting TimelapseCamera system")
        camera = TimelapseCamera()
        camera.run()
    except KeyboardInterrupt:
        logger.info("Shutting down due to keyboard interrupt")
        if 'camera' in locals():
            camera.cleanup()
    except Exception as e:
        logger.critical(f"Critical error in main: {str(e)}", exc_info=True)