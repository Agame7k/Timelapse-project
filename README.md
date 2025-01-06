# Timelapse Camera System (Beta)

## Overview
This project provides a sophisticated camera monitoring system with Discord integration, motion detection, and automated image capture capabilities. The system is currently in early beta stages.

## Features
- 🎥 Multi-camera support with automatic cycling (WIP)
- 🤖 Discord bot integration for remote control
- 📷 Motion detection with customizable sensitivity
- 🔄 Automated timelapse generation
- 💾 Dual storage system (local + network backup)
- ⚙️ Extensive configuration options
- 🛠️ Maintenance mode for system calibration
- 📊 Detailed status monitoring and reporting

## Requirements
- Python 3.8+
- Discord.py library
- OpenCV (cv2)
- SMB support for network storage
- Connected webcams or camera devices
- Linux or Windows OS

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/timelapse2.0.git
cd timelapse2.0
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create `cred.env` file with required credentials:
```env
# Discord Configuration
DISCORD_TOKEN=your_bot_token
GUILD_ID=your_server_id
COMMAND_CHANNEL_ID=your_channel_id
MOTION_ALERT_CHANNEL_ID=your_alert_channel_id

# Network Storage
SMB_HOST=your_nas_ip
SMB_USERNAME=your_username
SMB_PASSWORD=your_password
SMB_SHARE=your_share_name
SMB_PATH=/path/to/storage
```

## Discord Bot Setup

1. Visit [Discord Developer Portal](https://discord.com/developers/applications)
2. Create a new application
3. Go to "Bot" section and create a bot
4. Enable required intents:
    - Message Content Intent
    - Server Members Intent
5. Copy the bot token to your `cred.env` file
6. Use OAuth2 URL Generator to invite bot to your server:
    - Select "bot" scope
    - Select required permissions:
      - Send Messages
      - Embed Links
      - Attach Files
      - Read Message History
      - Add Reactions

## Usage

1. Start the system:
```bash
python timelapse.py
```

2. Discord Commands:
```
/capture - Take images from all cameras
/timelapse - Generate timelapse video
/status - View system status
/config - View current configuration
/upload - Upload to network storage
/pull - Download from network storage
/maintenance - Toggle maintenance mode
/notifications - Toggle motion alerts
```

## Directory Structure
```
timelapse2.0/
├── timelapse.py
├── cred.env
├── README.md
├── timelapse_photos_primary/
│   └── camera0/
├── timelapse_photos_backup/
│   └── camera0/
└── logs/
```

## Configuration

Default settings can be modified in the `Config` class:
- `COOLDOWN_SECONDS`: Minimum time between captures
- `MOTION_THRESHOLD`: Sensitivity for motion detection
- `CAMERA_INDEXES`: List of camera device indexes
- `FPS`: Output framerate for timelapses

## Known Issues
- Camera cycling may need manual restart occasionally
- Network storage reconnection can be unstable
- Motion detection sensitivity needs fine-tuning
- Limited error recovery in early beta

## License
MIT License - See LICENSE file for details

## Beta Notice ⚠️
This project is in early beta stages. Expect bugs and incomplete features. Use in production environments at your own risk.
