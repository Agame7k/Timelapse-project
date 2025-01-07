# main.py
import os
import time
from datetime import datetime
import logging
from dotenv import load_dotenv
import discord
from discord import app_commands
import asyncio
import cv2
from smb.SMBConnection import SMBConnection
from PIL import Image
import io
import sys
import glob

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
        self.start_time = time.time()
        self.last_capture_time = 0
        self.capture_interval = 60
        
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
                    status=discord.Status.dnd,
                    type=discord.ActivityType.watching,
                    name="ROBOTS IN THE MAKING"
                )
                await self.discord_client.change_presence(activity=activity)
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
                        f"Notifications: {'Enabled' if self.notification_enabled else 'Disabled'}\n"
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

        @self.tree.command(name="toggle-motion", description="Toggle motion monitoring")
        async def toggle_monitoring(interaction: discord.Interaction):
            self.is_monitoring = not self.is_monitoring
            status = "Enabled" if self.is_monitoring else "Disabled"
            
            # Update Discord status with DND when monitoring, Idle when not
            await self.discord_client.change_presence(
                status=discord.Status.dnd if self.is_monitoring else discord.Status.idle,
                activity=discord.Activity(
                    type=discord.ActivityType.watching if self.is_monitoring else discord.ActivityType.listening,
                    name="ROBOTS IN THE MAKING" if self.is_monitoring else "cause im blind"
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
                # Import required modules for system info
                import psutil
                import platform
                from datetime import datetime
                
                # System Platform Info
                system_info = {
                    'system': platform.system(),
                    'node': platform.node(),
                    'release': platform.release(),
                    'version': platform.version(),
                    'machine': platform.machine(),
                    'processor': platform.processor()
                }
                
                # CPU Info
                cpu_info = {
                    'usage': psutil.cpu_percent(interval=1),
                    'cores': psutil.cpu_count(logical=False),
                    'threads': psutil.cpu_count(logical=True),
                    'freq': psutil.cpu_freq().current if hasattr(psutil.cpu_freq(), 'current') else 0
                }
                
                # Memory Info
                memory = psutil.virtual_memory()
                memory_info = {
                    'total': memory.total,
                    'available': memory.available,
                    'used': memory.used,
                    'percent': memory.percent
                }
                
                # Disk Info
                disk = psutil.disk_usage('/')
                disk_info = {
                    'total': disk.total,
                    'used': disk.used,
                    'free': disk.free,
                    'percent': disk.percent
                }
                
                # Count files in capture directories
                total_captures = 0
                capture_folders = glob.glob('captures_*')
                for folder in capture_folders:
                    if os.path.isdir(folder):
                        total_captures += len([f for f in os.listdir(folder) if f.endswith('.jpg')])
                
                # Count timelapse photos
                timelapse_photos = 0
                if os.path.exists('timelapse_photos_primary'):
                    timelapse_photos = len([f for f in os.listdir('timelapse_photos_primary') if f.endswith('.jpg')])
                
                # Get system uptime
                system_uptime = datetime.now() - datetime.fromtimestamp(psutil.boot_time())
                bot_uptime = time.time() - self.start_time
                
                # Create rich embed
                embed = discord.Embed(
                    title="🖥️ System Status Dashboard",
                    description="Comprehensive system status and monitoring information",
                    color=discord.Color.blue(),
                    timestamp=datetime.now()
                )
                
                # System Information Section
                embed.add_field(
                    name="🔧 System Information",
                    value=f"```\n"
                        f"OS: {system_info['system']} {system_info['release']}\n"
                        f"Host: {system_info['node']}\n"
                        f"Architecture: {system_info['machine']}\n"
                        f"Python: {platform.python_version()}\n"
                        f"```",
                    inline=False
                )
                
                # CPU Status
                cpu_status = "🟢" if cpu_info['usage'] < 70 else "🟡" if cpu_info['usage'] < 90 else "🔴"
                embed.add_field(
                    name=f"{cpu_status} CPU Status",
                    value=f"```\n"
                        f"Usage: {cpu_info['usage']}%\n"
                        f"Cores: {cpu_info['cores']} ({cpu_info['threads']} threads)\n"
                        f"Frequency: {cpu_info['freq']/1000:.1f} GHz\n"
                        f"```",
                    inline=True
                )
                
                # Memory Status
                mem_status = "🟢" if memory_info['percent'] < 70 else "🟡" if memory_info['percent'] < 90 else "🔴"
                embed.add_field(
                    name=f"{mem_status} Memory Status",
                    value=f"```\n"
                        f"Used: {memory_info['used']/1024**3:.1f}GB\n"
                        f"Free: {memory_info['available']/1024**3:.1f}GB\n"
                        f"Total: {memory_info['total']/1024**3:.1f}GB\n"
                        f"```",
                    inline=True
                )
                
                # Storage Status
                storage_status = "🟢" if disk_info['percent'] < 70 else "🟡" if disk_info['percent'] < 90 else "🔴"
                embed.add_field(
                    name=f"{storage_status} Storage Status",
                    value=f"```\n"
                        f"Used: {disk_info['used']/1024**3:.1f}GB\n"
                        f"Free: {disk_info['free']/1024**3:.1f}GB\n"
                        f"Total: {disk_info['total']/1024**3:.1f}GB\n"
                        f"```",
                    inline=True
                )
                
                # Camera Status Section
                embed.add_field(
                    name="📸 Camera Status",
                    value=f"```\n"
                        f"Monitoring: {'Active' if self.is_monitoring else 'Paused'}\n"
                        f"Notifications: {'Enabled' if self.notifications_enabled else 'Disabled'}\n"
                        f"Last Motion: {self.format_last_motion_time()}\n"
                        f"Capture Count: {total_captures:,} images\n"
                        f"Timelapse Photos: {timelapse_photos:,} images\n"
                        f"```",
                    inline=False
                )
                
                # Uptime Information
                hours, remainder = divmod(int(bot_uptime), 3600)
                minutes, seconds = divmod(remainder, 60)
                sys_days = system_uptime.days
                sys_hours, remainder = divmod(system_uptime.seconds, 3600)
                sys_minutes, sys_seconds = divmod(remainder, 60)
                
                embed.add_field(
                    name="⏰ Uptime Information",
                    value=f"```\n"
                        f"Bot: {hours}h {minutes}m {seconds}s\n"
                        f"System: {sys_days}d {sys_hours}h {sys_minutes}m\n"
                        f"```",
                    inline=False
                )
                
                # Add footer with refresh info
                embed.set_footer(text=f"Status refreshed at {datetime.now().strftime('%H:%M:%S')} • Use /status to refresh")
                
                await interaction.followup.send(embed=embed)
                logger.info(f"Detailed status requested by {interaction.user}")
                
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
            start_time = time.time()
            
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
                value=f"{(time.time() - start_time):.2f}s",
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
            
            confirm_embed.add_field(
                name="⚠️ Warning",
                value="This will:\n• Stop all motion monitoring\n• Close camera connections\n• End Discord integration\n• Shutdown the bot process",
                inline=False
            )
            
            confirm_embed.set_footer(text="Click Confirm to proceed with shutdown")

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
                # Create shutdown embed
                shutdown_embed = discord.Embed(
                    title="🔌 System Shutdown Initiated",
                    description="Beginning shutdown sequence...",
                    color=discord.Color.red(),
                    timestamp=datetime.now()
                )

                shutdown_embed.add_field(
                    name="📊 Final Status",
                    value=f"""```
        Uptime: {self.format_uptime()}
        Total Captures: {sum(len(files) for _, _, files in os.walk('captures_'))} files
        Last Motion: {self.format_last_motion_time()}
        Camera Status: {'Connected' if self.camera.isOpened() else 'Disconnected'}```""",
                    inline=False
                )

                shutdown_embed.add_field(
                    name="👤 Triggered By",
                    value=interaction.user.mention,
                    inline=True
                )

                shutdown_embed.add_field(
                    name="⏰ Timestamp",
                    value=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    inline=True
                )

                # Add system resource usage
                import psutil
                process = psutil.Process()
                shutdown_embed.add_field(
                    name="🖥️ Resource Usage",
                    value=f"""```
        CPU: {psutil.cpu_percent()}%
        Memory: {process.memory_percent():.1f}%
        Threads: {process.num_threads()}
        Handle Count: {process.num_handles() if os.name == 'nt' else 'N/A'}```""",
                    inline=False
                )

                shutdown_embed.set_footer(text="System shutting down... Goodbye! 👋")
                
                # Disable all buttons
                for child in view.children:
                    child.disabled = True
                
                await interaction.response.edit_message(embed=shutdown_embed, view=view)
                
                # Log shutdown
                logger.info(f"System shutdown initiated by {interaction.user}")
                
                # Cleanup and exit
                self.cleanup()
                await self.discord_client.close()
                os._exit(0)

            async def cancel_callback(interaction: discord.Interaction):
                cancel_embed = discord.Embed(
                    title="✅ Shutdown Cancelled",
                    description="System will continue running normally",
                    color=discord.Color.green(),
                    timestamp=datetime.now()
                )
                
                # Disable all buttons
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

            # Send initial confirmation message
            await interaction.response.send_message(embed=confirm_embed, view=view)

            # Handle timeout
            async def on_timeout():
                timeout_embed = discord.Embed(
                    title="⏰ Shutdown Cancelled",
                    description="Confirmation timed out after 30 seconds",
                    color=discord.Color.greyple(),
                    timestamp=datetime.now()
                )
                
                # Disable all buttons
                for child in view.children:
                    child.disabled = True
                    
                await interaction.edit_original_response(embed=timeout_embed, view=view)

            view.on_timeout = on_timeout
            
        @self.tree.command(name="create_timelapse", description="Create timelapse from motion captures")
        async def create_timelapse(
            interaction: discord.Interaction,
            fps: int = 30,
            date: str = None
        ):
            await interaction.response.defer()
            
            embed = discord.Embed(
                title="🎬 Creating Timelapse",
                description="Processing captured images...",
                color=discord.Color.blue(),
                timestamp=datetime.now()
            )
            embed.add_field(
                name="FPS",
                value=str(fps),
                inline=True
            )
            embed.add_field(
                name="Date",
                value=date if date else "Today",
                inline=True
            )
            
            video_path = await self.create_timelapse_video(fps, date)
            if video_path:
                try:
                    await self.upload_to_smb(video_path)
                    embed.title = "✅ Timelapse Created"
                    embed.description = "Video has been created and uploaded successfully"
                    embed.color = discord.Color.green()
                    embed.add_field(
                        name="Output",
                        value=f"`{video_path}`",
                        inline=False
                    )
                except Exception as e:
                    embed.title = "⚠️ Upload Failed"
                    embed.description = "Video created but upload failed"
                    embed.color = discord.Color.yellow()
                    embed.add_field(
                        name="Error",
                        value=str(e),
                        inline=False
                    )
            else:
                embed.title = "❌ Timelapse Failed"
                embed.description = "Failed to create timelapse video"
                embed.color = discord.Color.red()
            
            await interaction.followup.send(embed=embed)

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
            embed.set_footer(text="System will reboot in 10 seconds")
            
            await interaction.response.send_message(embed=embed)
            logger.info(f"System reboot initiated by {interaction.user}")
            
            # Create reboot script
            reboot_script = """#!/bin/bash
        sleep 10
        sudo reboot
        rm -- "$0"
        """
            with open("reboot_pi.sh", "w") as f:
                f.write(reboot_script)
            
            # Make script executable and run
            os.chmod("reboot_pi.sh", 0o755)
            
            # Cleanup and execute reboot
            self.cleanup()
            await self.discord_client.close()
            os.system("./reboot_pi.sh &")
            sys.exit(0)

        @self.tree.command(name="sync", description="Sync files between Pi and SMB server")
        async def sync_files(interaction: discord.Interaction):
            await interaction.response.defer()
            start_time = time.time()
            
            embed = discord.Embed(
                title="🔄 Starting Sync",
                description="Synchronizing files between Pi and SMB server...",
                color=discord.Color.blue(),
                timestamp=datetime.now()
            )
            await interaction.followup.send(embed=embed)
            
            try:
                # Connect to SMB
                if not self.smb_client.connect(
                    os.getenv('SMB_SERVER_IP'),
                    int(os.getenv('SMB_PORT', 445))
                ):
                    raise Exception("Failed to connect to SMB server")

                share_name = os.getenv('SMB_SHARE_NAME')
                local_dirs = ['captures_*', 'timelapse_photos_primary']
                stats = {
                    'uploaded': 0,
                    'downloaded': 0,
                    'skipped': 0,
                    'errors': 0
                }

                # Sync each directory
                for dir_pattern in local_dirs:
                    for local_dir in glob.glob(dir_pattern):
                        if not os.path.isdir(local_dir):
                            continue

                        # Create remote directory if it doesn't exist
                        try:
                            self.smb_client.createDirectory(share_name, local_dir)
                        except:
                            pass

                        # Upload local files
                        local_files = {f: os.path.getmtime(os.path.join(local_dir, f))
                                    for f in os.listdir(local_dir)
                                    if os.path.isfile(os.path.join(local_dir, f))}

                        # Get remote files
                        remote_files = {}
                        try:
                            file_list = self.smb_client.listPath(share_name, local_dir)
                            for f in file_list:
                                if f.isRegular:
                                    remote_files[f.filename] = f.last_write_time
                        except:
                            logger.error(f"Failed to list remote directory: {local_dir}")
                            continue

                        # Upload new/modified local files
                        for fname, local_mtime in local_files.items():
                            try:
                                if fname not in remote_files or local_mtime > remote_files[fname]:
                                    local_path = os.path.join(local_dir, fname)
                                    with open(local_path, 'rb') as file:
                                        if self.smb_client.storeFile(share_name, f"{local_dir}/{fname}", file):
                                            stats['uploaded'] += 1
                                            logger.info(f"Uploaded: {local_path}")
                                else:
                                    stats['skipped'] += 1
                            except Exception as e:
                                logger.error(f"Error uploading {fname}: {str(e)}")
                                stats['errors'] += 1

                        # Download new remote files
                        for fname, _ in remote_files.items():
                            try:
                                local_path = os.path.join(local_dir, fname)
                                if fname not in local_files:
                                    with open(local_path, 'wb') as file:
                                        if self.smb_client.retrieveFile(share_name, f"{local_dir}/{fname}", file):
                                            stats['downloaded'] += 1
                                            logger.info(f"Downloaded: {local_path}")
                            except Exception as e:
                                logger.error(f"Error downloading {fname}: {str(e)}")
                                stats['errors'] += 1

                # Create summary embed
                elapsed_time = time.time() - start_time
                embed = discord.Embed(
                    title="✅ Sync Complete",
                    description="File synchronization completed",
                    color=discord.Color.green(),
                    timestamp=datetime.now()
                )
                embed.add_field(
                    name="📤 Uploaded",
                    value=str(stats['uploaded']),
                    inline=True
                )
                embed.add_field(
                    name="📥 Downloaded",
                    value=str(stats['downloaded']),
                    inline=True
                )
                embed.add_field(
                    name="⏭️ Skipped",
                    value=str(stats['skipped']),
                    inline=True
                )
                if stats['errors'] > 0:
                    embed.add_field(
                        name="⚠️ Errors",
                        value=str(stats['errors']),
                        inline=True
                    )
                embed.add_field(
                    name="⏱️ Time Taken",
                    value=f"{elapsed_time:.1f}s",
                    inline=True
                )
                embed.set_footer(text="Check logs for detailed information")

                await interaction.followup.send(embed=embed)
                logger.info(f"Sync completed in {elapsed_time:.1f}s")

            except Exception as e:
                error_embed = discord.Embed(
                    title="❌ Sync Failed",
                    description=str(e),
                    color=discord.Color.red(),
                    timestamp=datetime.now()
                )
                await interaction.followup.send(embed=error_embed)
                logger.error(f"Sync failed: {str(e)}")
            finally:
                try:
                    self.smb_client.close()
                except:
                    pass
        
        @self.tree.command(name="clear_cache", description="Clear all locally captured photos")
        async def clear_cache(interaction: discord.Interaction):
            """Clear all locally captured photos"""
            if not interaction.user.guild_permissions.administrator:
                embed = discord.Embed(
                    title="❌ Permission Denied",
                    description="Only administrators can clear the cache",
                    color=discord.Color.red()
                )
                await interaction.response.send_message(embed=embed, ephemeral=True)
                return
                
            await interaction.response.defer()
            start_time = time.time()
            
            try:
                deleted_count = 0
                total_size = 0
                
                # Find and clear all capture folders
                capture_folders = glob.glob('captures_*')
                
                for folder in capture_folders:
                    if os.path.isdir(folder):
                        # Calculate folder size before deletion
                        folder_size = sum(os.path.getsize(os.path.join(folder, f)) 
                                        for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)))
                        total_size += folder_size
                        
                        # Delete folder and count files
                        deleted_count += len(os.listdir(folder))
                        import shutil
                        shutil.rmtree(folder)
                        logger.info(f"Deleted folder: {folder}")
                        
                embed = discord.Embed(
                    title="🧹 Cache Cleared",
                    description="All local capture folders have been removed",
                    color=discord.Color.green(),
                    timestamp=datetime.now()
                )
                embed.add_field(
                    name="📊 Statistics",
                    value=f"Deleted: {deleted_count} files\nSpace freed: {total_size / (1024*1024):.2f} MB",
                    inline=False
                )
                embed.add_field(
                    name="⏱️ Time Taken",
                    value=f"{(time.time() - start_time):.2f}s",
                    inline=True
                )
                embed.add_field(
                    name="👤 Triggered By",
                    value=interaction.user.mention,
                    inline=True
                )
                embed.set_footer(text="All local captures have been cleared")
                
                await interaction.followup.send(embed=embed)
                logger.info(f"Cache cleared by {interaction.user} - {deleted_count} files deleted")
                
            except Exception as e:
                error_embed = discord.Embed(
                    title="❌ Cache Clear Failed",
                    description=str(e),
                    color=discord.Color.red()
                )
                await interaction.followup.send(embed=error_embed)
                logger.error(f"Cache clear failed: {str(e)}")

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

    def format_uptime(self):
        """Format the system uptime"""
        uptime = time.time() - self.start_time
        hours, remainder = divmod(int(uptime), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours}h {minutes}m {seconds}s"

    def format_last_motion_time(self):
        """Format the last motion time for status display"""
        if self.last_motion_time == 0:
            return "No motion detected yet"
        time_diff = time.time() - self.last_motion_time
        return f"{int(time_diff)} seconds ago"

    async def send_snapshot(self, interaction):
        """Take and send a snapshot directly to Discord"""
        logger.info("Taking snapshot")
        start_time = time.time()
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
        elapsed_time = time.time() - start_time
        logger.info(f"Snapshot captured and sent in {elapsed_time:.2f}s")

    def detect_motion(self, frame):
        """Detect motion in frame with cooldown period and return contours"""
        current_time = time.time()
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
        start_time = time.time()
        current_time = time.time()
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
                    value=f"{(time.time() - self.last_motion_time):.2f}s",
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
            
            elapsed_time = time.time() - start_time
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
        """Take scheduled snapshots every 30 minutes"""
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
            await asyncio.sleep(1800)  # 30 minutes = 1800 seconds

    async def run_camera(self):
        """Run the camera monitoring"""
        try:
            await self.monitor()
        except Exception as e:
            logger.error(f"Error in camera monitoring: {str(e)}")

    def run(self):
        """Run the system"""
        try:
            self.start_time = time.time()
            loop = asyncio.get_event_loop()
            
            # Create initialization task
            init_task = loop.create_task(self.initialize())
            
            # Create other tasks
            loop.create_task(self.run_camera())
            loop.create_task(self.scheduled_capture())
            
            # Start Discord client
            loop.run_until_complete(self.discord_client.start(os.getenv('DISCORD_TOKEN')))
            
        except KeyboardInterrupt:
            self.cleanup()
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            self.cleanup()

    def cleanup(self):
        """Cleanup resources"""
        logger.info("Starting cleanup")
        try:
            if hasattr(self, 'camera') and self.camera.isOpened():
                self.camera.release()
                logger.info("Camera released")
            cv2.destroyAllWindows()
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