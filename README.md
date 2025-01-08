# Timelapse Camera Monitoring System

A Python-based camera monitoring system with Discord integration, motion detection, and automated timelapse creation capabilities.
> **⚠️ Note:**
> This script was developed for a specific robotics timelapse setup and will likely need modifications to suit your requirements. The code is provided as-is and may require adjustments for different environments or use cases.

## Features

- 📸 Real-time motion detection and capture
- 🤖 Discord bot integration with command interface
- 🎥 Automated timelapse creation
- 📁 SMB file sharing support
- ⏱️ Scheduled capture every 30 minutes
- 🔔 Configurable Discord notifications
- 📊 System monitoring and statistics
- 📝 Timesheet tracking and statistics (with reminder at your choice (mine is 1:25 PM)

## Requirements

- Python 3.8+
- OpenCV (cv2)
- Discord.py
- SMB support
- Raspberry Pi (recommended) or system with camera

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Agame7k/Timelapse-project.git
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create `cred.env` file with your credentials:
```env
DISCORD_TOKEN=your_discord_token
DISCORD_CHANNEL_ID=your_channel_id
SMB_USERNAME=your_smb_username
SMB_PASSWORD=your_smb_password
SMB_SERVER_IP=your_server_ip
SMB_PORT=445
SMB_SHARE_NAME=your_share_name
CLIENT_NAME=your_client_name
SERVER_NAME=your_server_name
OWNER_ID=your_discord_id
```

## Discord Commands

- `/status` - View system status and statistics
- `/snapshot` - Take and send immediate snapshot
- `/toggle-motion` - Enable/disable motion detection
- `/create_timelapse` - Create timelapse from captures
- `/sync` - Sync files with SMB server
- `/notifications` - Toggle Discord notifications
- `/reboot_pi` - Reboot the Raspberry Pi system
- `/kill` - Safely shutdown the system
- `/start_clock` - starts time tracking
- `/stop_clock` - stops time tracking
- `/timesheet_stats` - view timesheet statistics

## Usage

Run the script:
```bash
python Timelapse.py
```

## Logging

The system logs all activities to `timelapse.log` including:

- System startup/shutdown
- Motion detection events
- File operations
- Error messages

## License

This project is licensed under the MIT License - see the LICENSE file for details.
