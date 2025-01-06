import os
import time
from datetime import datetime, timedelta
import cv2
import numpy as np
from dotenv import load_dotenv
import discord
from discord import app_commands
import asyncio
import schedule
import threading
from pathlib import Path
import logging
from smb.SMBConnection import SMBConnection
import glob
from moviepy import ImageSequenceClip
import io
import platform
import signal
import json
import math

# Configuration
class Config:
    """Configuration class for the timelapse application."""
    def __init__(self):
        self.COOLDOWN_SECONDS = 30
        self.MOTION_THRESHOLD = 2
        self.CAMERA_INDEXES = [0, 2]
        self.LOCAL_SAVE_PATH_1 = "timelapse_photos_primary/"
        self.LOCAL_SAVE_PATH_2 = "timelapse_photos_backup/"
        self.FPS = 24

        

        # Load environment variables with defaults
        env_path = Path('cred.env')
        if not env_path.exists():
            raise FileNotFoundError("cred.env file not found. Please create it using the template.")
            
        load_dotenv(dotenv_path=env_path)
        
        # Required variables with validation
        self.SMB_HOST = os.getenv('SMB_HOST', '')
        self.SMB_USERNAME = os.getenv('SMB_USERNAME', '')
        self.SMB_PASSWORD = os.getenv('SMB_PASSWORD', '')
        self.SMB_SHARE = os.getenv('SMB_SHARE', '')
        self.SMB_PATH = os.getenv('SMB_PATH', '/')
        self.DISCORD_TOKEN = os.getenv('DISCORD_TOKEN', '')
        
        # Convert IDs to integers with validation
        try:
            self.GUILD_ID = int(os.getenv('GUILD_ID', '0'))
            self.COMMAND_CHANNEL_ID = int(os.getenv('COMMAND_CHANNEL_ID', '0'))
            self.MOTION_ALERT_CHANNEL_ID = int(os.getenv('MOTION_ALERT_CHANNEL_ID', '0'))
        except ValueError:
            raise ValueError("GUILD_ID, COMMAND_CHANNEL_ID, and MOTION_ALERT_CHANNEL_ID must be valid integers")

        # Validate required variables
        if not all([self.SMB_HOST, self.GUILD_ID, self.COMMAND_CHANNEL_ID]):
            missing = []
            if not self.SMB_HOST: missing.append('SMB_HOST')
            if not self.GUILD_ID: missing.append('GUILD_ID')
            if not self.COMMAND_CHANNEL_ID: missing.append('COMMAND_CHANNEL_ID')
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

        self.MOTION_ALERT_COOLDOWN = 5 # 5 minutes between motion alerts


    def get_camera_path(self, base_path, camera_index):
        """Get camera-specific directory path"""
        return os.path.join(base_path, f"camera{camera_index}")

DEFAULT_CAMERA_SETTINGS = {
    'resolution_options': [
        (1280, 720),   # HD
        (1920, 1080),  # Full HD
        (2560, 1440),  # 2K
        (3840, 2160)   # 4K
    ],
    'default_settings': {
        'resolution': (1280, 720),
        'fps': 30,
        'brightness': 50,
        'contrast': 50,
        'saturation': 60,
        'exposure': -6,
        'gain': 0
    },
    'parameter_ranges': {
        'brightness': (0, 100),
        'contrast': (-50, 50),
        'saturation': (0, 100),
        'exposure': (-10, 0),
        'gain': (0, 100)
    }
}

# Setup logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('timelapse.log'),
        logging.StreamHandler()
    ]
)
class Camera:
    def __init__(self, camera_index, bot=None):
        self.camera_index = camera_index
        self.cap = None
        self.last_frame = None
        self.last_capture_time = 0
        self.motion_value = 0
        self.is_monitoring = False
        self.monitor_thread = None
        self.bot = bot
        self.last_motion_alert = 0  
        self.notifications_enabled = True
        self.last_active_time = 0
        self.cycle_interval = 30  # Seconds between camera switches
        self.motion_check_interval = 0.5  # Seconds between motion checks
        self.last_motion_check = 0
        self.consecutive_failures = 0
        self.backoff_time = 1  # Initial backoff in seconds
        self.max_backoff = 300  # Maximum backoff of 5 minutes
        self.camera_index = camera_index
        self.cap = None

        

        self.settings = {
            'resolution': (1280, 720),  # (width, height)
            'fps': 15,
            'brightness': 50,
            'contrast': 50,
            'saturation': 60,
            'exposure': -6
        }
        self.default_settings = None
        self.settings_file = f"camera_{camera_index}_settings.json"
        self.settings_file = f"camera_{camera_index}_settings.json"


    def should_be_active(self, other_cameras):
        """Determine if this camera should be active based on cycling"""
        current_time = time.time()
        
        # If no cameras are active, activate this one
        if not any(cam.cap and cam.cap.isOpened() for cam in other_cameras):
            if not self.cap or not self.cap.isOpened():
                asyncio.create_task(self.initialize())
            return True
            
        # If this camera was recently active, check if it's time to switch
        if self.last_active_time > 0:
            if current_time - self.last_active_time >= self.cycle_interval:
                self.last_active_time = 0
                self.deinitialize()  # Full shutdown when switching
                return False
            return True
            
        # Check if other cameras are due for cycling
        for cam in other_cameras:
            if cam.last_active_time > 0 and current_time - cam.last_active_time < cam.cycle_interval:
                if self.cap:  # Always deinitialize if another camera is active
                    self.deinitialize()
                return False
        
        # Initialize this camera if it's now active
        if not self.cap or not self.cap.isOpened():
            asyncio.create_task(self.initialize())
        return True


    def load_or_create_settings(self):
        """Load existing settings or create new ones"""
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    self.settings = json.load(f)
                    if self.validate_settings(self.settings):
                        logging.info(f"Loaded existing settings for camera {self.camera_index}")
                        return True
            
            # Create new settings file
            self.settings = DEFAULT_CAMERA_SETTINGS['default_settings'].copy()
            self.save_settings()
            logging.info(f"Created new settings for camera {self.camera_index}")
            return True
            
        except Exception as e:
            logging.error(f"Error managing settings for camera {self.camera_index}: {e}")
            self.settings = DEFAULT_CAMERA_SETTINGS['default_settings'].copy()
            return False

    def validate_settings(self, settings):
        """Validate camera settings against allowed ranges"""
        try:
            required_keys = ['resolution', 'fps', 'brightness', 'contrast', 'saturation', 'exposure']
            if not all(key in settings for key in required_keys):
                return False
                
            ranges = DEFAULT_CAMERA_SETTINGS['parameter_ranges']
            
            # Validate each parameter
            for param, (min_val, max_val) in ranges.items():
                if param in settings:
                    if not min_val <= settings[param] <= max_val:
                        settings[param] = DEFAULT_CAMERA_SETTINGS['default_settings'][param]
                        
            # Validate resolution
            if not isinstance(settings['resolution'], (list, tuple)) or len(settings['resolution']) != 2:
                settings['resolution'] = DEFAULT_CAMERA_SETTINGS['default_settings']['resolution']
                
            return True
            
        except Exception as e:
            logging.error(f"Settings validation error: {e}")
            return False

    async def initialize(self):
        """Modified initialize with settings reload"""
        try:
            # Always start fresh
            if self.cap:
                self.deinitialize()
                
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                self.consecutive_failures += 1
                logging.error(f"Failed to open camera {self.camera_index} (Attempt {self.consecutive_failures})")
                return False
                
            # Reset failure count on success    
            self.consecutive_failures = 0
            self.backoff_time = 1
                
            # Load settings fresh each time
            self.settings = DEFAULT_CAMERA_SETTINGS['default_settings'].copy()
            if os.path.exists(self.settings_file):
                try:
                    with open(self.settings_file, 'r') as f:
                        loaded_settings = json.load(f)
                        if self.validate_settings(loaded_settings):
                            self.settings = loaded_settings
                except:
                    logging.warning(f"Failed to load settings for camera {self.camera_index}, using defaults")
            
            # Try resolutions with fallback
            resolutions = [
                (1280, 720),
                (800, 600), 
                (640, 480)
            ]
            
            success = False
            for res in resolutions:
                self.settings['resolution'] = res
                if self.apply_settings():
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        success = True
                        logging.info(f"Camera {self.camera_index} initialized at {res[0]}x{res[1]}")
                        break
                await asyncio.sleep(0.5)
                
            if not success:
                self.deinitialize()
                return False
                
            return True
            
        except Exception as e:
            logging.error(f"Camera initialization error: {e}")
            if self.cap:
                self.deinitialize()
            return False
        
    def deinitialize(self):
        """Fully shut down and clean up camera resources"""
        try:
            if self.monitor_thread and self.monitor_thread.is_alive():
                self.stop_monitoring()
            
            if self.cap and self.cap.isOpened():
                self.cap.release()
            
            self.cap = None
            self.last_frame = None
            self.last_active_time = 0
            self.last_motion_check = 0
            self.motion_value = 0
            logging.info(f"Camera {self.camera_index} fully deinitialized")
            
        except Exception as e:
            logging.error(f"Error deinitializing camera {self.camera_index}: {e}")

    def apply_settings(self):
        """Apply current settings to camera with validation"""
        if not self.cap or not self.cap.isOpened():
            return False
            
        try:
            # Resolution must be set first
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.settings['resolution'][0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.settings['resolution'][1])
            
            # Verify resolution was set
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if abs(actual_width - self.settings['resolution'][0]) > 50 or \
            abs(actual_height - self.settings['resolution'][1]) > 50:
                logging.warning(f"Resolution mismatch for camera {self.camera_index}")
                
            # Apply other settings
            self.cap.set(cv2.CAP_PROP_FPS, self.settings['fps'])
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.settings['brightness'])
            self.cap.set(cv2.CAP_PROP_CONTRAST, self.settings['contrast'])
            self.cap.set(cv2.CAP_PROP_SATURATION, self.settings['saturation'])
            self.cap.set(cv2.CAP_PROP_EXPOSURE, self.settings['exposure'])
            
            # Verify stream is still valid
            ret, frame = self.cap.read()
            return ret and frame is not None
            
        except Exception as e:
            logging.error(f"Error applying settings: {e}")
            return False

    def save_settings(self):
        """Save current settings to file"""
        try:
            if self.validate_settings(self.settings):
                with open(self.settings_file, 'w') as f:
                    json.dump(self.settings, f, indent=4)
                return True
        except Exception as e:
            logging.error(f"Error saving settings: {e}")
        return False

    def get_supported_resolutions(self):
        """Test and get supported resolutions"""
        supported = []
        for width, height in DEFAULT_CAMERA_SETTINGS['resolution_options']:
            if self.test_resolution(width, height):
                supported.append((width, height))
        return supported or [(1280, 720)]  # Fallback to HD

    def test_resolution(self, width, height):
        """Test if a resolution is supported"""
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Check if resolution was set within tolerance
            if abs(actual_w - width) <= 50 and abs(actual_h - height) <= 50:
                ret, frame = self.cap.read()
                return ret and frame is not None
                
            return False
            
        except:
            return False

    @classmethod
    async def create(cls, camera_index, bot=None):
        """Async factory method to create and initialize a camera"""
        camera = cls(camera_index, bot)
        camera.bot = bot  # Ensure bot is set
        await camera.initialize()
        return camera
    
    async def calibrate_camera(self):
        """Calibrate camera with progress tracking"""
        if not self.cap.isOpened():
            yield {
                'phase': 'Error',
                'message': 'Camera not open',
                'success': False,
                'progress': 0,
                'total_steps': 1,
                'current_step': 0
            }
            return

        try:
            # Store original settings
            original_settings = {
                'resolution': (
                    int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                ),
                'fps': int(self.cap.get(cv2.CAP_PROP_FPS)),
                'brightness': int(self.cap.get(cv2.CAP_PROP_BRIGHTNESS)),
                'contrast': int(self.cap.get(cv2.CAP_PROP_CONTRAST)),
                'saturation': int(self.cap.get(cv2.CAP_PROP_SATURATION)),
                'exposure': int(self.cap.get(cv2.CAP_PROP_EXPOSURE))
            }

            # Test resolutions
            resolutions = DEFAULT_CAMERA_SETTINGS['resolution_options']
            best_settings = None
            best_score = 0
            total_steps = len(resolutions)
            current_step = 0

            for width, height in resolutions:
                current_step += 1
                yield {
                    'phase': 'Testing Resolution',
                    'message': f'Testing {width}x{height}',
                    'success': True,
                    'progress': (current_step / total_steps) * 100,
                    'total_steps': total_steps,
                    'current_step': current_step,
                    'resolution': f'{width}x{height}'
                }

                if not self.test_resolution(width, height):
                    continue

                # Test image quality
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    continue

                quality_score = self._calculate_quality_metrics(frame)
                
                if quality_score > best_score:
                    best_score = quality_score
                    best_settings = {
                        'resolution': (width, height),
                        'fps': original_settings['fps'],
                        'brightness': original_settings['brightness'],
                        'contrast': original_settings['contrast'],
                        'saturation': original_settings['saturation'],
                        'exposure': original_settings['exposure'],
                        'quality_score': quality_score,
                        'last_calibrated': datetime.now().isoformat()
                    }

                    yield {
                        'phase': 'New Best Found',
                        'message': f'New best settings found (Score: {quality_score:.2f})',
                        'success': True,
                        'progress': (current_step / total_steps) * 100,
                        'total_steps': total_steps,
                        'current_step': current_step,
                        'settings': best_settings
                    }

            if best_settings:
                # Save and apply best settings
                self.settings = best_settings
                self.save_settings()
                self.apply_settings()
                
                yield {
                    'phase': 'Complete',
                    'message': f'Calibration complete (Score: {best_score:.2f})',
                    'success': True,
                    'progress': 100,
                    'total_steps': total_steps,
                    'current_step': total_steps,
                    'settings': best_settings
                }
            else:
                # Restore original settings
                self._restore_settings(original_settings)
                yield {
                    'phase': 'Failed',
                    'message': 'No improvement found, restored original settings',
                    'success': False,
                    'progress': 100,
                    'total_steps': total_steps,
                    'current_step': total_steps
                }

        except Exception as e:
            self._restore_settings(original_settings)
            yield {
                'phase': 'Error',
                'message': f'Calibration error: {str(e)}',
                'success': False,
                'progress': 0,
                'total_steps': 1,
                'current_step': 0
            }

    def _calculate_quality_metrics(self, frame):
        """Calculate comprehensive image quality score"""
        try:
            # Convert to different color spaces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Brightness metrics
            brightness_mean = np.mean(gray)
            brightness_std = np.std(gray)
            brightness_score = 1.0 - abs(brightness_mean - 128) / 128
            
            # Color metrics
            saturation_mean = np.mean(hsv[:,:,1])
            saturation_std = np.std(hsv[:,:,1])
            color_score = (saturation_mean / 255.0) * (saturation_std / 128.0)
            
            # Contrast metrics
            contrast_score = brightness_std / 128.0
            
            # Edge/sharpness metrics
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness_score = np.std(laplacian) / 128.0
            
            # Check for overexposure
            overexposed = np.mean(gray > 240)
            overexposure_penalty = 1.0 - overexposed
            
            # Calculate final score (weighted average)
            weights = {
                'brightness': 0.3,
                'color': 0.3,
                'contrast': 0.2,
                'sharpness': 0.1,
                'overexposure': 0.1
            }
            
            final_score = (
                weights['brightness'] * brightness_score +
                weights['color'] * color_score +
                weights['contrast'] * contrast_score +
                weights['sharpness'] * sharpness_score +
                weights['overexposure'] * overexposure_penalty
            )
            
            return min(max(final_score, 0.0), 1.0)  # Clamp between 0 and 1
            
        except Exception as e:
            logging.error(f"Error calculating quality metrics: {e}")
            return 0.0

    def _test_resolution(self, width, height):
        """Test if a resolution is supported with validation"""
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            
            # Verify resolution was set correctly
            actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Allow small tolerance in resolution
            width_ok = abs(actual_w - width) <= 50
            height_ok = abs(actual_h - height) <= 50
            
            if not (width_ok and height_ok):
                return False
                
            # Verify stream is still valid
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return False
                
            return True
            
        except:
            return False

    def _restore_settings(self, settings):
        """Restore camera to previous settings"""
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, settings['resolution'][0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, settings['resolution'][1])
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, settings['brightness'])
            self.cap.set(cv2.CAP_PROP_CONTRAST, settings['contrast'])
            self.cap.set(cv2.CAP_PROP_SATURATION, settings['saturation'])
            self.cap.set(cv2.CAP_PROP_EXPOSURE, settings['exposure'])
            self.cap.set(cv2.CAP_PROP_FPS, settings['fps'])
        except Exception as e:
            logging.error(f"Error restoring settings: {e}")

    def toggle_notifications(self):
        self.notifications_enabled = not self.notifications_enabled
        return self.notifications_enabled

    def start_monitoring(self):
        """Start motion detection monitoring"""
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            if not self.cap.isOpened():
                logging.error(f"Cannot start monitoring - Camera {self.camera_index} is not open")
                return False
                
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._motion_loop, daemon=True)
            self.monitor_thread.start()
            logging.info(f"Started motion monitoring for camera {self.camera_index}")
            return True
        return False
    
    def verify_monitoring(self):
        """Verify motion detection is running"""
        if not self.is_monitoring or not self.monitor_thread or not self.monitor_thread.is_alive():
            logging.warning(f"Motion detection stopped for camera {self.camera_index} - attempting restart")
            return self.start_monitoring()
        return True

    def stop_monitoring(self):
        self.is_monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join()
            logging.info(f"Stopped motion monitoring for camera {self.camera_index}")

    def _motion_loop(self):
        """Modified motion detection loop with camera cycling"""
        while self.is_monitoring:
            current_time = time.time()
            
            # Check if this camera should be active
            if not self.should_be_active([cam for cam in self.bot.cameras if cam != self]):
                asyncio.sleep(1)  # Sleep if not active
                continue
                
            # Ensure camera is initialized
            if not self.cap or not self.cap.isOpened():
                success = self.initialize()
                if not success:
                    asyncio.sleep(self.backoff_time)
                    continue
            
            # Update active time
            self.last_active_time = current_time
                
            # Rate limit motion checks
            if current_time - self.last_motion_check < self.motion_check_interval:
                time.sleep(0.1)  # Small sleep to prevent CPU thrashing
                continue
                
            try:
                motion_detected, frame, contours = self.detect_motion()
                self.last_motion_check = current_time
                
                if motion_detected and frame is not None:
                    self.capture_image()
                    if self.bot:
                        frame_with_boxes = self.draw_motion_boxes(frame, contours)
                        asyncio.run_coroutine_threadsafe(
                            self.send_motion_alert(frame_with_boxes), 
                            self.bot.loop
                        )
                        
            except Exception as e:
                logging.error(f"Motion detection error: {e}")
                time.sleep(1)  # Error backoff
                
            # Yield to other cameras
            asyncio.sleep(0.1)
        
    def draw_motion_boxes(self, frame, contours):
        """Draw tech-styled skeletal visualization of detected objects"""
        frame_with_boxes = frame.copy()
        height, width = frame.shape[:2]
        
        if not contours:
            return frame_with_boxes

        # Join nearby contours
        joined_contours = []
        used = set()
        
        for i, c1 in enumerate(contours):
            if i in used:
                continue
                
            x1, y1, w1, h1 = cv2.boundingRect(c1)
            if w1 * h1 < 5000:  # Minimum size filter
                continue
                
            merged = c1
            for j, c2 in enumerate(contours[i+1:], i+1):
                if j in used:
                    continue
                x2, y2, w2, h2 = cv2.boundingRect(c2)
                if abs(x1-x2) < w1 and abs(y1-y2) < h1:
                    merged = np.concatenate((merged, c2))
                    used.add(j)
            joined_contours.append(merged)

        # Process each detected object
        for idx, contour in enumerate(joined_contours):
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Find key points of the object
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
                
            # Calculate center and extremum points
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
            
            # Get extremum points
            leftmost = tuple(contour[contour[:,:,0].argmin()][0])
            rightmost = tuple(contour[contour[:,:,0].argmax()][0])
            topmost = tuple(contour[contour[:,:,1].argmin()][0])
            bottommost = tuple(contour[contour[:,:,1].argmax()][0])
            
            # Choose color with pulsing effect
            alliance = (255, 50, 50) if idx % 2 == 0 else (50, 50, 255)
            pulse = abs(math.sin(time.time() * 3))
            line_color = tuple(int(c * pulse) for c in alliance)
            
            # Draw skeleton structure
            # Center crosshair
            cv2.line(frame_with_boxes, 
                    (center_x-10, center_y), (center_x+10, center_y),
                    line_color, 2)
            cv2.line(frame_with_boxes,
                    (center_x, center_y-10), (center_x, center_y+10),
                    line_color, 2)
                    
            # Connect key points
            skeleton_points = [
                (leftmost, (center_x, center_y)),
                (rightmost, (center_x, center_y)),
                (topmost, (center_x, center_y)),
                (bottommost, (center_x, center_y))
            ]
            
            # Draw skeleton lines with data flow
            for start, end in skeleton_points:
                cv2.line(frame_with_boxes, start, end, line_color, 2)
                
                # Add moving data points
                t = (time.time() * 2) % 1
                point = (
                    int(start[0] + (end[0] - start[0]) * t),
                    int(start[1] + (end[1] - start[1]) * t)
                )
                cv2.circle(frame_with_boxes, point, 2, (0, 255, 255), -1)
            
            # Add pulsing nodes at key points
            for point in [leftmost, rightmost, topmost, bottommost]:
                node_pulse = abs(math.sin(time.time() * 4 + hash(point) % 4))
                cv2.circle(frame_with_boxes, point, 
                        int(4 + node_pulse * 3), line_color, 2)
                        
            # Add tech overlay
            cv2.putText(frame_with_boxes,
                    f"TARGET {idx}",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    line_color, 2)
                    
            # Object dimensions
            dims = f"W:{w}px H:{h}px"
            cv2.putText(frame_with_boxes, dims,
                    (x + w + 5, center_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    line_color, 1)

        return frame_with_boxes
    
    def detect_motion(self):
        if not self.cap.isOpened():
            logging.error(f"Camera {self.camera_index} is not open")
            return False, None, []  # Return empty list instead of None for contours
            
        ret, frame = self.cap.read()
        if not ret:
            logging.error(f"Failed to read frame from camera {self.camera_index}")
            return False, None, []

        
        # Convert to grayscale and blur
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        if self.last_frame is None:
            self.last_frame = gray
            return False, frame, None
            
        # Calculate frame delta and threshold
        frame_delta = cv2.absdiff(self.last_frame, gray)
        thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        # Find contours
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Calculate overall motion value
        self.motion_value = np.mean(thresh)
        motion_detected = self.motion_value > config.MOTION_THRESHOLD

        logging.info(f"Camera {self.camera_index} - Motion: {self.motion_value:.2f}")
        
        self.last_frame = gray
        return motion_detected, frame, contours
        
    def capture_image(self, force=False):
        if hasattr(self.bot, 'maintenance_mode') and self.bot.maintenance_mode and not force:
            logging.info(f"Camera {self.camera_index} - Capture skipped (Maintenance Mode)")
            return None, None
        
        current_time = time.time()
        
        if not force and (current_time - self.last_capture_time) < config.COOLDOWN_SECONDS:
            return None, None
            
        motion, frame, _ = self.detect_motion()
        if motion or force:
            if frame is None:
                return None, None
                
            # Create camera-specific directories
            camera_dir = config.get_camera_path(config.LOCAL_SAVE_PATH_1, self.camera_index)
            backup_dir = config.get_camera_path(config.LOCAL_SAVE_PATH_2, self.camera_index)
            os.makedirs(camera_dir, exist_ok=True)
            os.makedirs(backup_dir, exist_ok=True)
                
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"camera_{self.camera_index}_{timestamp}.jpg"
            local_path = os.path.join(camera_dir, filename)
            backup_path = os.path.join(backup_dir, filename)
            
            cv2.imwrite(local_path, frame)  # Save to primary
            cv2.imwrite(backup_path, frame)  # Save to backup
            self.last_capture_time = current_time
            logging.info(f"Captured image: {filename}")
            return local_path, frame
        return None, None

    async def send_motion_alert(self, frame):
        """Send a professional motion detection alert with embedded image and metrics."""
        if not self.bot or not frame.any() or not self.notifications_enabled:
            return
        current_time = time.time()
        if current_time - self.last_motion_alert < config.MOTION_ALERT_COOLDOWN:
            return

        try:
            channel = self.bot.get_channel(config.MOTION_ALERT_CHANNEL_ID)
            if channel:
                # Get fresh motion detection with contours
                motion, new_frame, contours = self.detect_motion()
                if motion and new_frame is not None:
                    # Draw boxes on the frame for the alert
                    frame_with_boxes = self.draw_motion_boxes(new_frame, contours)
                    _, buffer = cv2.imencode('.jpg', frame_with_boxes)
                    file = discord.File(io.BytesIO(buffer.tobytes()), 
                                    filename=f"motion_alert_{self.camera_index}.jpg")
                    
                    # Create modern embed
                    embed = discord.Embed(
                        title=f"{self.bot.EMOJIS['motion']} Motion Alert",
                        description=(
                            f"**Camera {self.camera_index}** detected significant motion\n"
                            f"Time: <t:{int(current_time)}:R>"
                        ),
                        color=discord.Color.from_rgb(255, 75, 75),  # Vibrant red
                        timestamp=datetime.now()
                    )

                    # Add detection metrics
                    embed.add_field(
                        name="📊 Detection Metrics",
                        value=(
                            f"Motion Level: `{self.motion_value:.2f}`\n"
                            f"Threshold: `{config.MOTION_THRESHOLD}`\n"
                            f"Areas Detected: `{len([c for c in contours if cv2.contourArea(c) > 500])}`"
                        ),
                        inline=True
                    )

                    # Add camera status
                    resolution = (
                        int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    )
                    embed.add_field(
                        name="🎥 Camera Status",
                        value=(
                            f"Resolution: `{resolution[0]}x{resolution[1]}`\n"
                            f"Frame Rate: `{int(self.cap.get(cv2.CAP_PROP_FPS))} FPS`\n"
                            f"Quality: `{self._calculate_quality_metrics(new_frame):.2f}`"
                        ),
                        inline=True
                    )

                    # Add the motion image as main embed image
                    embed.set_image(url=f"attachment://motion_alert_{self.camera_index}.jpg")

                    # Add helpful footer
                    embed.set_footer(
                        text=(
                            f"Motion Detection System • Camera {self.camera_index} • "
                            f"Use /notifications to manage alerts"
                        )
                    )

                    await channel.send(embed=embed, file=file)
                    self.last_motion_alert = current_time
                    logging.info(f"Sent enhanced motion alert for camera {self.camera_index}")

        except Exception as e:
            logging.error(f"Failed to send motion alert: {e}")

class SMBUploader:
    def __init__(self):
        self.conn = None
        self.last_connection_attempt = 0
        self.connection_retry_delay = 300  # 5 minutes
        self.is_connected = False
        self.connect()  # Try initial connection
        
    def connect(self):
        current_time = time.time()
        
        if (current_time - self.last_connection_attempt) < self.connection_retry_delay:
            return self.is_connected
            
        self.last_connection_attempt = current_time
        
        try:
            if self.conn:
                try:
                    self.conn.close()
                except:
                    pass
                    
            self.conn = SMBConnection(
                config.SMB_USERNAME,
                config.SMB_PASSWORD,
                "TimeLapseClient",
                config.SMB_HOST,
                use_ntlm_v2=True
            )
            
            self.is_connected = self.conn.connect(config.SMB_HOST, 445, timeout=30)
            
            if self.is_connected:
                try:
                    self.conn.listPath(config.SMB_SHARE, '/')
                    logging.info("Successfully connected to SMB server and validated share access")
                except Exception as e:
                    self.is_connected = False
                    logging.error(f"Share access validation failed: {e}")
            
            return self.is_connected
            
        except Exception as e:
            self.is_connected = False
            logging.error(f"SMB connection failed: {e}")
            return False
            
    def check_connection(self):
        if not self.is_connected:
            return self.connect()
            
        try:
            self.conn.echo('echo')
            return True
        except:
            self.is_connected = False
            return self.connect()

    def upload_folder(self, local_path):
        """Upload all files from a folder to SMB share, organizing by camera number"""
        if not os.path.exists(local_path):
            logging.warning(f"Folder not found: {local_path}")
            return False

        if not self.check_connection():
            logging.error("Cannot upload - SMB connection unavailable")
            return False

        success = True
        try:
            for filename in os.listdir(local_path):
                if not filename.endswith('.jpg'):
                    continue
                    
                # Extract camera number from filename (format: camera_X_timestamp.jpg)
                try:
                    camera_num = filename.split('_')[1]
                    remote_path = f"camera{camera_num}/"
                except IndexError:
                    logging.warning(f"Skipping file with invalid format: {filename}")
                    continue

                local_file = os.path.join(local_path, filename)
                try:
                    # Create remote directory if it doesn't exist
                    try:
                        self.conn.listPath(config.SMB_SHARE, remote_path)
                    except:
                        self.conn.createDirectory(config.SMB_SHARE, remote_path)

                    # Upload file to camera-specific folder
                    remote_file = f"{remote_path}{filename}"
                    with open(local_file, 'rb') as file:
                        self.conn.storeFile(config.SMB_SHARE, remote_file, file)
                        logging.info(f"Uploaded {filename} to {remote_path}")
                except Exception as e:
                    logging.error(f"Failed to upload {filename}: {e}")
                    success = False
                    
            return success
        except Exception as e:
            logging.error(f"Folder upload failed: {e}")
            return False
        
    def pull_images(self, remote_camera_path, local_path):
        """Pull images from SMB share to local directory"""
        if not self.check_connection():
            logging.error("Cannot pull - SMB connection unavailable")
            return False
            
        success = True
        try:
            # List files in remote camera directory
            remote_files = self.conn.listPath(config.SMB_SHARE, remote_camera_path)
            
            # Ensure local directory exists
            os.makedirs(local_path, exist_ok=True)
            
            for file_info in remote_files:
                if not file_info.filename.endswith('.jpg'):
                    continue
                    
                remote_path = f"{remote_camera_path}/{file_info.filename}"
                local_file_path = os.path.join(local_path, file_info.filename)
                
                # Skip if file already exists locally
                if os.path.exists(local_file_path):
                    continue
                    
                # Download file to camera-specific directory
                with open(local_file_path, 'wb') as file:
                    self.conn.retrieveFile(config.SMB_SHARE, remote_path, file)
                    logging.info(f"Downloaded {file_info.filename} to {local_path}")
                    
            return success
        except Exception as e:
            logging.error(f"Pull operation failed: {e}")
            return False
        
class TimelapseGenerator:
    @staticmethod
    async def create_timelapse(camera_index, fps=None):
        if fps is None:
            fps = config.FPS
            
        # Get list of images from camera-specific directory
        camera_dir = config.get_camera_path(config.LOCAL_SAVE_PATH_1, camera_index)
        pattern = os.path.join(camera_dir, f"camera_{camera_index}_*.jpg")
        images = sorted(glob.glob(pattern))
        
        if not images:
            return None, "No images found for the specified camera."
            
        # Create timelapse
        try:
            clip = ImageSequenceClip(images, fps=fps)
            output_path = f"timelapse_camera_{camera_index}_{int(time.time())}.mp4"
            clip.write_videofile(output_path, codec='libx264', audio=False)
            return output_path, None
        except Exception as e:
            return None, f"Error creating timelapse: {str(e)}"

class DiscordBot(discord.Client):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.tree = app_commands.CommandTree(self)
        self.cameras = []
        self.smb_uploader = None
        self.maintenance_mode = False
        self.start_time = datetime.now()
        self.shutting_down = False
        self.shutdown_event = asyncio.Event()
        self._ready = asyncio.Event() 
        self.setup_commands()
        
        # Add emoji constants
        self.EMOJIS = {
            'success': '✅',
            'error': '❌',
            'warning': '⚠️',
            'camera': '📸',
            'video': '🎥',
            'upload': '📤',
            'status': '📊',
            'config': '⚙️',
            'motion': '🔄',
            'connected': '🟢',
            'disconnected': '🔴', 
            'storage': '💾',
            'download': '📥',
            'maintenance': '🔧',
            'active': '⚡'
        }

    async def shutdown(self):
        """Graceful shutdown of bot and services"""
        if self.shutting_down:
            return
        self.shutting_down = True
        
        try:
            # Set status to offline first
            await self.change_presence(
                status=discord.Status.offline,
                activity=None
            )
            
            # Send shutdown notice
            if hasattr(self, 'cameras'):
                channel = self.get_channel(config.COMMAND_CHANNEL_ID)
                if channel:
                    shutdown_embed = discord.Embed(
                        title=f"{self.EMOJIS['warning']} System Shutdown",
                        description="System is performing a graceful shutdown...",
                        color=discord.Color.orange(),
                        timestamp=datetime.now()
                    )
                    shutdown_embed.add_field(
                        name="Status",
                        value="• Setting bot status to offline\n"
                            "• Stopping camera monitoring\n"
                            "• Closing network connections\n"
                            "• Saving system state",
                        inline=False
                    )
                    await channel.send(embed=shutdown_embed)
            
            # Stop cameras
            if hasattr(self, 'cameras'):
                for camera in self.cameras:
                    camera.stop_monitoring()
                logging.info("Stopped all camera monitoring")
            
            # Close SMB connection
            if hasattr(self, 'smb_uploader'):
                if self.smb_uploader.conn:
                    self.smb_uploader.conn.close()
                logging.info("Closed SMB connection")
            
            # Set shutdown event
            self.shutdown_event.set()
            
            # Close Discord connection
            await self.close()
            logging.info("Closed Discord connection")
            
        except Exception as e:
            logging.error(f"Error during shutdown: {e}")
            # Try one last time to set offline status
            try:
                await self.change_presence(status=discord.Status.offline)
            except:
                pass
        finally:
            logging.info("Shutdown complete")

    def update_dependencies(self, cameras=None, smb_uploader=None):
        if cameras is not None:
            self.cameras = cameras
        if smb_uploader is not None:
            self.smb_uploader = smb_uploader
        
    def setup_commands(self):

        @self.tree.command(
            name="capture",
            description="Capture an image from all cameras"
        )
        async def capture(interaction: discord.Interaction):
            await self.capture_command(interaction)

        @self.tree.command(
            name="timelapse",
            description="Generate a timelapse video"
        )
        async def timelapse(interaction: discord.Interaction, camera_index: int):
            await self.timelapse_command(interaction, camera_index)

        # Similarly update other commands
        @self.tree.command(
            name="upload",
            description="Upload pending images"
        )
        async def upload(interaction: discord.Interaction):
            await self.upload_command(interaction)

        @self.tree.command(
            name="pull",
            description="Pull images from network storage"
        )
        async def pull(interaction: discord.Interaction):
            await self.pull_command(interaction)

        @self.tree.command(
            name="status",
            description="Show system status"
        )
        async def status(interaction: discord.Interaction):
            await self.status_command(interaction)

        @self.tree.command(
            name="notifications",
            description="Toggle motion notifications"
        )
        async def notifications(interaction: discord.Interaction):
            await self.toggle_notifications_command(interaction)

        @self.tree.command(
            name="config",
            description="Show configuration"
        )
        async def config(interaction: discord.Interaction):
            await self.config_command(interaction)

        @self.tree.command(
            name="reboot",
            description="Reboot the system (Linux only)"
        )
        @app_commands.default_permissions(administrator=True)
        async def reboot(interaction: discord.Interaction):
            await self.reboot_command(interaction)

        @self.tree.command(
            name="kill",
            description="Shut down the bot and all services"
        )
        @app_commands.default_permissions(administrator=True)
        async def kill(interaction: discord.Interaction):
            await self.kill_command(interaction)
            
        @self.tree.command(
            name="maintenance",
            description="Toggle system maintenance mode"
        )
        @app_commands.default_permissions(administrator=True)
        async def maintenance(interaction: discord.Interaction):
            await self.maintenance_command(interaction)

    async def setup_hook(self):
        """Async setup that runs before the bot starts"""
        logging.info("Bot setup hook running")
        # Don't sync commands here - wait for on_ready
        
    async def initialize_bot(self):
        """Called after bot is ready to perform additional setup"""
        try:
            logging.info("Starting bot initialization sequence...")
            
            # Set status
            await self.change_presence(
                status=discord.Status.dnd,
                activity=discord.Activity(
                    type=discord.ActivityType.watching,
                    name="RI3D"
                )
            )
            
            # Initialize services only after bot is ready
            cameras, smb_uploader = await initialize_services()
            
            # Update bot with dependencies
            self.update_dependencies(cameras=cameras, smb_uploader=smb_uploader)
            
            # Verify motion detection is running
            for camera in self.cameras:
                if not camera.verify_monitoring():
                    logging.error(f"Failed to start motion detection for camera {camera.camera_index}")
            
            # Start scheduled tasks thread
            schedule_thread = threading.Thread(
                target=setup_scheduled_tasks,
                args=(cameras, smb_uploader),
                daemon=True
            )
            schedule_thread.start()
            
            # Sync commands to guild
            await self.tree.sync()
            logging.info("Command tree synced")
            
            # Signal ready state
            self._ready.set()
            logging.info("Bot initialization complete")
            
        except Exception as e:
            logging.error(f"Error in initialization: {e}")
            raise

    async def on_ready(self):
        """Called when bot is ready"""
        try:
            logging.info(f"Bot connected as {self.user}")
            await self.initialize_bot()
        except Exception as e:
            logging.error(f"Error in on_ready: {e}")
            raise

    def signal_handler(self, sig):
        """Handle shutdown signals with logging"""
        logging.info(f"Received signal {sig}")
        # Create event loop if needed
        if not asyncio.get_event_loop().is_running():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        asyncio.get_event_loop().create_task(self.shutdown())

    async def capture_command(self, interaction):
    # Channel permission validation
        if interaction.channel_id != config.COMMAND_CHANNEL_ID:
            error_embed = discord.Embed(
                title=f"{self.EMOJIS['error']} Unauthorized Channel",
                description="Please use the designated command channel for this operation.",
                color=discord.Color.red()
            ).set_footer(text="Access Denied")
            await interaction.response.send_message(embed=error_embed, ephemeral=True)
            return

        await interaction.response.defer()

        # Initial status embed
        status_embed = discord.Embed(
            title=f"{self.EMOJIS['camera']} Capture Operation",
            description="Initiating multi-camera capture sequence...",
            color=discord.Color.blue(),
            timestamp=datetime.now()
        ).add_field(
            name="Operation Status",
            value=f"{self.EMOJIS['motion']} Preparing cameras...",
            inline=False
        ).set_footer(text="Processing capture requests...")
        status_message = await interaction.followup.send(embed=status_embed)

        results = []
        total_cameras = len(self.cameras)
        successful_captures = 0

        for index, camera in enumerate(self.cameras, 1):
            # Create per-camera embed
            capture_embed = discord.Embed(
                title=f"{self.EMOJIS['camera']} Camera {camera.camera_index}",
                description=f"Processing camera {index}/{total_cameras}",
                color=discord.Color.blue(),
                timestamp=datetime.now()
            )

            # Attempt capture
            path, frame = camera.capture_image(force=True)
            
            if path and frame is not None:
                # Successful capture
                successful_captures += 1
                _, buffer = cv2.imencode('.jpg', frame)
                file = discord.File(
                    io.BytesIO(buffer.tobytes()), 
                    filename=f"camera_{camera.camera_index}_capture.jpg"
                )
                
                # Add image as thumbnail
                capture_embed.set_thumbnail(url="attachment://camera_{camera.camera_index}_capture.jpg")
                
                capture_embed.color = discord.Color.green()
                capture_embed.add_field(
                    name=f"{self.EMOJIS['success']} Status",
                    value="Image captured successfully",
                    inline=True
                )
                capture_embed.add_field(
                    name="📏 Resolution",
                    value=f"`{frame.shape[1]}x{frame.shape[0]}`",
                    inline=True
                )
                capture_embed.add_field(
                    name="📂 Filename",
                    value=f"`{os.path.basename(path)}`",
                    inline=True
                )
                capture_embed.add_field(
                    name="📊 Image Details",
                    value=f"Size: `{buffer.size / 1024:.1f}KB`\nFormat: `JPEG`",
                    inline=False
                )
                
                results.append((capture_embed, file))
            else:
                # Failed capture
                capture_embed.color = discord.Color.red()
                capture_embed.add_field(
                    name=f"{self.EMOJIS['error']} Status",
                    value="Capture failed",
                    inline=True
                )
                capture_embed.add_field(
                    name="Reason",
                    value="Camera not responding or busy",
                    inline=True
                )
                results.append((capture_embed, None))

            # Update progress
            progress = int((index / total_cameras) * 20)
            progress_bar = "█" * progress + "░" * (20 - progress)
            status_embed.set_field_at(
                0, 
                name="Operation Status",
                value=(
                    f"Processing: `{progress_bar}` {index}/{total_cameras}\n"
                    f"Successful: `{successful_captures}`\n"
                    f"Current: Camera {camera.camera_index}"
                ),
                inline=False
            )
            await status_message.edit(embed=status_embed)

        # Send individual results
        for embed, file in results:
            if file:
                await interaction.channel.send(embed=embed, file=file)
            else:
                await interaction.channel.send(embed=embed)

        # Final summary embed
        summary_embed = discord.Embed(
            title=f"{self.EMOJIS['camera']} Capture Operation Complete",
            description="All cameras processed",
            color=discord.Color.green() if successful_captures == total_cameras 
                else discord.Color.orange() if successful_captures > 0 
                else discord.Color.red(),
            timestamp=datetime.now()
        ).add_field(
            name="📊 Results Summary",
            value=(
                f"Total Cameras: `{total_cameras}`\n"
                f"Successful Captures: `{successful_captures}`\n"
                f"Failed Captures: `{total_cameras - successful_captures}`"
            ),
            inline=False
        ).add_field(
            name="💾 Storage Location",
            value=f"```\n{config.LOCAL_SAVE_PATH_1}\n```",
            inline=False
        ).set_footer(text="Operation completed")

        await interaction.channel.send(embed=summary_embed)

    async def toggle_notifications_command(self, interaction):
        """
        Toggle motion detection notifications for all cameras.
        Provides detailed status feedback and camera-specific information.
        """
        # Channel permission validation
        if interaction.channel_id != config.COMMAND_CHANNEL_ID:
            error_embed = discord.Embed(
                title=f"{self.EMOJIS['error']} Unauthorized Channel",
                description=(
                    "Please use the designated command channel for this operation.\n"
                    f"Channel ID: `{config.COMMAND_CHANNEL_ID}`"
                ),
                color=discord.Color.red(),
                timestamp=datetime.now()
            ).set_footer(text="Access Denied")
            await interaction.response.send_message(embed=error_embed, ephemeral=True)
            return

        # Validate camera availability
        if not self.cameras:
            error_embed = discord.Embed(
                title=f"{self.EMOJIS['error']} No Cameras Available",
                description="No cameras are currently connected to the system.",
                color=discord.Color.red(),
                timestamp=datetime.now()
            )
            await interaction.response.send_message(embed=error_embed, ephemeral=True)
            return

        # Toggle notifications state
        new_state = not self.cameras[0].notifications_enabled
        
        # Create status embed
        status_embed = discord.Embed(
            title=f"{self.EMOJIS['camera']} Motion Detection Notifications",
            description=(
                f"Notifications have been **{'enabled' if new_state else 'disabled'}** for all cameras\n\n"
                f"{'🔔' if new_state else '🔕'} Global Notification State: `{'Active' if new_state else 'Inactive'}`"
            ),
            color=discord.Color.green() if new_state else discord.Color.red(),
            timestamp=datetime.now()
        )

        # Update cameras and build detailed status report
        camera_status = []
        cameras_online = 0
        
        for camera in self.cameras:
            camera.notifications_enabled = new_state
            is_online = camera.cap.isOpened()
            if is_online:
                cameras_online += 1
                
            # Build detailed camera status
            status = (
                f"**Camera {camera.camera_index}**\n"
                f"{self.EMOJIS['connected'] if is_online else self.EMOJIS['disconnected']} "
                f"Status: `{'Online' if is_online else 'Offline'}`\n"
                f"{'🔔' if new_state else '🔕'} Notifications: "
                f"`{'Enabled' if new_state else 'Disabled'}`\n"
                f"{'🔍' if camera.is_monitoring else '⏸️'} Monitoring: "
                f"`{'Active' if camera.is_monitoring else 'Paused'}`\n"
                f"📊 Motion Level: `{camera.motion_value:.2f}`"
            )
            camera_status.append(status)

        # Add system overview field
        status_embed.add_field(
            name="System Overview",
            value=(
                f"Total Cameras: `{len(self.cameras)}`\n"
                f"Online Cameras: `{cameras_online}`\n"
                f"System Health: {self.EMOJIS['success'] if cameras_online == len(self.cameras) else self.EMOJIS['warning']}"
            ),
            inline=False
        )

        # Add individual camera status field
        status_embed.add_field(
            name="Camera Status",
            value="\n\n".join(camera_status),
            inline=False
        )

        # Add configuration info
        status_embed.add_field(
            name="Configuration",
            value=(
                f"Motion Threshold: `{config.MOTION_THRESHOLD}`\n"
                f"Alert Cooldown: `{config.MOTION_ALERT_COOLDOWN}` minutes\n"
                f"Alert Channel: <#{config.MOTION_ALERT_CHANNEL_ID}>"
            ),
            inline=False
        )

        # Add action summary
        status_embed.add_field(
            name="Action Summary",
            value=(
                f"{self.EMOJIS['success']} Successfully {'enabled' if new_state else 'disabled'} "
                f"notifications for {len(self.cameras)} camera(s)\n"
                f"Time: `{datetime.now().strftime('%H:%M:%S')}`"
            ),
            inline=False
        )

        # Add helpful footer
        status_embed.set_footer(
            text=(
                f"Use /status for detailed system information | "
                f"Motion alerts will be sent to #{config.MOTION_ALERT_CHANNEL_ID}"
            )
        )

        await interaction.response.send_message(embed=status_embed)

    async def timelapse_command(self, interaction, camera_index):
        """Generate a timelapse video with detailed progress tracking and status information."""
        # Channel permission validation
        if interaction.channel_id != config.COMMAND_CHANNEL_ID:
            error_embed = discord.Embed(
                title=f"{self.EMOJIS['error']} Unauthorized Channel",
                description="Please use the designated command channel for this operation.",
                color=discord.Color.red(),
                timestamp=datetime.now()
            ).set_footer(text="Access Denied")
            await interaction.response.send_message(embed=error_embed, ephemeral=True)
            return

        # Validate camera index
        if camera_index not in config.CAMERA_INDEXES:
            error_embed = discord.Embed(
                title=f"{self.EMOJIS['error']} Invalid Camera",
                description=f"Camera {camera_index} is not configured in the system.\nAvailable cameras: `{', '.join(map(str, config.CAMERA_INDEXES))}`",
                color=discord.Color.red(),
                timestamp=datetime.now()
            )
            await interaction.response.send_message(embed=error_embed, ephemeral=True)
            return

        await interaction.response.defer()

        # Initial status embed
        status_embed = discord.Embed(
            title=f"{self.EMOJIS['video']} Timelapse Generation",
            description=(
                f"Preparing timelapse for Camera {camera_index}\n\n"
                f"🎥 Source: `Camera {camera_index}`\n"
                f"🖼️ FPS: `{config.FPS}`\n"
                f"📂 Source Path: `{config.LOCAL_SAVE_PATH_1}/camera{camera_index}`"
            ),
            color=discord.Color.blue(),
            timestamp=datetime.now()
        )
        status_embed.add_field(
            name="Initialization",
            value=f"{self.EMOJIS['motion']} Scanning for source images...",
            inline=False
        )
        status_msg = await interaction.followup.send(embed=status_embed)

        # Get image list and validate
        camera_dir = config.get_camera_path(config.LOCAL_SAVE_PATH_1, camera_index)
        pattern = os.path.join(camera_dir, f"camera_{camera_index}_*.jpg")
        images = sorted(glob.glob(pattern))
        
        if not images:
            error_embed = discord.Embed(
                title=f"{self.EMOJIS['warning']} No Images Found",
                description=(
                    f"No source images found for Camera {camera_index}\n\n"
                    f"Expected Path: `{camera_dir}`\n"
                    f"Pattern: `camera_{camera_index}_*.jpg`"
                ),
                color=discord.Color.orange(),
                timestamp=datetime.now()
            )
            error_embed.add_field(
                name="Possible Solutions",
                value="• Capture some images first using `/capture`\n"
                    "• Check camera directory permissions\n"
                    "• Verify camera is working properly",
                inline=False
            )
            await status_msg.edit(embed=error_embed)
            return

        # Update status with image count
        status_embed.add_field(
            name="Source Images",
            value=(
                f"Total Images: `{len(images)}`\n"
                f"Date Range: `{os.path.basename(images[0])[:-4]}` to\n"
                f"           `{os.path.basename(images[-1])[:-4]}`"
            ),
            inline=False
        )
        await status_msg.edit(embed=status_embed)

        # Generate timelapse
        try:
            # Progress updates
            status_embed.add_field(
                name="Generation Status",
                value=f"{self.EMOJIS['motion']} Creating video sequence...",
                inline=False
            )
            await status_msg.edit(embed=status_embed)

            clip = ImageSequenceClip(images, fps=config.FPS)
            output_filename = f"timelapse_camera_{camera_index}_{int(time.time())}.mp4"
            
            # Update status before encoding
            status_embed.set_field_at(
                -1,
                name="Generation Status",
                value=f"{self.EMOJIS['video']} Encoding video...\n"
                    f"Output: `{output_filename}`\n"
                    f"FPS: `{config.FPS}`\n"
                    f"Duration: `{len(images)/config.FPS:.1f}s`",
                inline=False
            )
            await status_msg.edit(embed=status_embed)

            # Generate video
            clip.write_videofile(
                output_filename,
                codec='libx264',
                audio=False,
                preset='medium',
                bitrate='8000k'
            )

            # Success embed
            success_embed = discord.Embed(
                title=f"{self.EMOJIS['success']} Timelapse Complete",
                description=(
                    f"Successfully generated timelapse for Camera {camera_index}\n\n"
                    f"📊 **Statistics**\n"
                    f"• Source Images: `{len(images)}`\n"
                    f"• Duration: `{len(images)/config.FPS:.1f}` seconds\n"
                    f"• FPS: `{config.FPS}`\n"
                    f"• Resolution: `{clip.size[0]}x{clip.size[1]}`\n"
                    f"• File Size: `{os.path.getsize(output_filename)/1024/1024:.1f}MB`"
                ),
                color=discord.Color.green(),
                timestamp=datetime.now()
            )
            success_embed.set_footer(text="Use /timelapse again to generate a new video")

            # Send final file
            await interaction.channel.send(
                embed=success_embed,
                file=discord.File(output_filename)
            )

            # Cleanup
            os.remove(output_filename)
            clip.close()

        except Exception as e:
            error_embed = discord.Embed(
                title=f"{self.EMOJIS['error']} Generation Failed",
                description=f"Failed to generate timelapse for Camera {camera_index}",
                color=discord.Color.red(),
                timestamp=datetime.now()
            )
            error_embed.add_field(
                name="Error Details",
                value=f"```\n{str(e)}\n```",
                inline=False
            )
            error_embed.add_field(
                name="Troubleshooting",
                value="• Check disk space\n"
                    "• Verify image file integrity\n"
                    "• Check system resources",
                inline=False
            )
            await status_msg.edit(embed=error_embed)

            # Log error
            logging.error(f"Timelapse generation failed for camera {camera_index}: {e}")

    async def upload_command(self, interaction):
        """Upload images to network storage with progress tracking"""
        # Validate channel permissions
        if interaction.channel_id != config.COMMAND_CHANNEL_ID:
            await interaction.response.send_message(
                embed=discord.Embed(
                    title=f"{self.EMOJIS['error']} Unauthorized Channel",
                    description="Please use the designated command channel for this operation.",
                    color=discord.Color.red(),
                    timestamp=datetime.now()
                ).set_footer(text="Access Denied"),
                ephemeral=True
            )
            return

        await interaction.response.defer()

        # Initial status embed
        status_embed = discord.Embed(
            title=f"{self.EMOJIS['upload']} Network Storage Upload",
            description="Preparing upload operation...",
            color=discord.Color.blue(),
            timestamp=datetime.now()
        ).set_footer(text="Scanning files...")
        status_msg = await interaction.followup.send(embed=status_embed)

        # Validate source directory
        if not os.path.exists(config.LOCAL_SAVE_PATH_1):
            await status_msg.edit(embed=discord.Embed(
                title=f"{self.EMOJIS['error']} Source Error",
                description="Source directory not found.",
                color=discord.Color.red(),
                timestamp=datetime.now()
            ))
            return

        # Get file list
        files = []
        for camera_index in config.CAMERA_INDEXES:
            camera_path = config.get_camera_path(config.LOCAL_SAVE_PATH_1, camera_index)
            if os.path.exists(camera_path):
                camera_files = [f for f in os.listdir(camera_path) if f.endswith('.jpg')]
                files.extend((camera_path, f) for f in camera_files)

        if not files:
            await status_msg.edit(embed=discord.Embed(
                title=f"{self.EMOJIS['warning']} No Files",
                description="No images found for upload.",
                color=discord.Color.orange(),
                timestamp=datetime.now()
            ))
            return

        # Verify network connection
        if not self.smb_uploader.check_connection():
            await status_msg.edit(embed=discord.Embed(
                title=f"{self.EMOJIS['error']} Network Error",
                description=(
                    "Failed to connect to network storage.\n\n"
                    f"Host: `{config.SMB_HOST}`\n"
                    f"Share: `{config.SMB_SHARE}`"
                ),
                color=discord.Color.red(),
                timestamp=datetime.now()
            ))
            return

        # Upload process
        total_files = len(files)
        uploaded = 0
        failed = 0
        
        for index, (local_path, filename) in enumerate(files, 1):
            try:
                # Extract camera number from filename
                camera_num = filename.split('_')[1]
                remote_path = f"camera{camera_num}/"
                local_file = os.path.join(local_path, filename)

                # Update progress
                progress = int((index / total_files) * 20)
                progress_bar = "█" * progress + "░" * (20 - progress)
                percentage = int((index / total_files) * 100)

                status_embed = discord.Embed(
                    title=f"{self.EMOJIS['upload']} Upload in Progress",
                    description=(
                        f"Uploading files to network storage...\n\n"
                        f"Current: `{filename}`\n"
                        f"Progress: `{progress_bar}` {percentage}%"
                    ),
                    color=discord.Color.blue(),
                    timestamp=datetime.now()
                )
                
                status_embed.add_field(
                    name="Statistics",
                    value=(
                        f"Total Files: `{total_files}`\n"
                        f"Uploaded: `{uploaded}`\n"
                        f"Failed: `{failed}`\n"
                        f"Remaining: `{total_files - (uploaded + failed)}`"
                    ),
                    inline=False
                )

                await status_msg.edit(embed=status_embed)

                # Ensure remote directory exists
                try:
                    self.smb_uploader.conn.listPath(config.SMB_SHARE, remote_path)
                except:
                    self.smb_uploader.conn.createDirectory(config.SMB_SHARE, remote_path)

                # Upload file
                with open(local_file, 'rb') as file:
                    self.smb_uploader.conn.storeFile(config.SMB_SHARE, f"{remote_path}{filename}", file)
                    uploaded += 1
                    logging.info(f"Uploaded {filename} to {remote_path}")

            except Exception as e:
                failed += 1
                logging.error(f"Failed to upload {filename}: {e}")

        # Final status
        final_embed = discord.Embed(
            title=f"{self.EMOJIS['upload']} Upload Operation Complete",
            description=(
                f"Network storage synchronization finished\n\n"
                f"Progress: `{'█' * 20}` 100%"
            ),
            color=(
                discord.Color.green() if failed == 0 else 
                discord.Color.orange() if uploaded > 0 else 
                discord.Color.red()
            ),
            timestamp=datetime.now()
        )

        final_embed.add_field(
            name="Results",
            value=(
                f"{self.EMOJIS['success']} Successfully uploaded: `{uploaded}`\n"
                f"{self.EMOJIS['error']} Failed uploads: `{failed}`\n"
                f"Total files processed: `{total_files}`"
            ),
            inline=False
        )

        if failed > 0:
            final_embed.add_field(
                name=f"{self.EMOJIS['warning']} Attention Required",
                value="Some files failed to upload. Check the system logs for details.",
                inline=False
            )

        final_embed.set_footer(text=f"Network: {config.SMB_HOST} | Share: {config.SMB_SHARE}")
        await status_msg.edit(embed=final_embed)

    async def pull_command(self, interaction):
        """Synchronize images from network storage"""
        # Validate channel permissions
        if interaction.channel_id != config.COMMAND_CHANNEL_ID:
            await interaction.response.send_message(
                embed=discord.Embed(
                    title=f"{self.EMOJIS['error']} Unauthorized Channel",
                    description="Please use the designated command channel for this operation.",
                    color=discord.Color.red(),
                    timestamp=datetime.now()
                ).set_footer(text="Access Denied"),
                ephemeral=True
            )
            return

        await interaction.response.defer()

        # Initial status embed
        status_embed = discord.Embed(
            title=f"{self.EMOJIS['download']} Network Storage Sync",
            description="Initializing synchronization process...",
            color=discord.Color.blue(),
            timestamp=datetime.now()
        ).set_footer(text="Checking connection...")
        status_msg = await interaction.followup.send(embed=status_embed)

        # Verify network connection
        if not self.smb_uploader.check_connection():
            await status_msg.edit(embed=discord.Embed(
                title=f"{self.EMOJIS['error']} Connection Failed",
                description=(
                    "Unable to connect to network storage.\n\n"
                    f"Host: `{config.SMB_HOST}`\n"
                    f"Share: `{config.SMB_SHARE}`"
                ),
                color=discord.Color.red(),
                timestamp=datetime.now()
            ))
            return

        # Process each camera
        total_cameras = len(config.CAMERA_INDEXES)
        success_count = 0
        total_files = 0
        downloaded_files = 0
        
        for index, camera_index in enumerate(config.CAMERA_INDEXES, 1):
            try:
                remote_path = f"camera{camera_index}"
                local_path = config.get_camera_path(config.LOCAL_SAVE_PATH_1, camera_index)
                os.makedirs(local_path, exist_ok=True)

                # Update progress
                progress = int((index / total_cameras) * 20)
                progress_bar = "█" * progress + "░" * (20 - progress)
                percentage = int((index / total_cameras) * 100)

                status_embed = discord.Embed(
                    title=f"{self.EMOJIS['download']} Sync in Progress",
                    description=(
                        f"Synchronizing Camera {camera_index}\n\n"
                        f"Progress: `{progress_bar}` {percentage}%"
                    ),
                    color=discord.Color.blue(),
                    timestamp=datetime.now()
                )

                status_embed.add_field(
                    name="Current Operation",
                    value=(
                        f"Processing: Camera {camera_index}\n"
                        f"Remote Path: `{remote_path}`\n"
                        f"Local Path: `{local_path}`"
                    ),
                    inline=False
                )

                # Add statistics field
                status_embed.add_field(
                    name="Statistics",
                    value=(
                        f"Cameras Processed: `{index}/{total_cameras}`\n"
                        f"Files Downloaded: `{downloaded_files}`\n"
                        f"Total Files Found: `{total_files}`"
                    ),
                    inline=False
                )

                await status_msg.edit(embed=status_embed)

                # List remote files
                remote_files = self.smb_uploader.conn.listPath(config.SMB_SHARE, remote_path)
                camera_files = [f for f in remote_files if f.filename.endswith('.jpg')]
                total_files += len(camera_files)

                # Download new files
                for file_info in camera_files:
                    remote_file = f"{remote_path}/{file_info.filename}"
                    local_file = os.path.join(local_path, file_info.filename)

                    if not os.path.exists(local_file):
                        with open(local_file, 'wb') as file:
                            self.smb_uploader.conn.retrieveFile(config.SMB_SHARE, remote_file, file)
                            downloaded_files += 1
                            logging.info(f"Downloaded {file_info.filename}")

                success_count += 1

            except Exception as e:
                logging.error(f"Failed to sync camera {camera_index}: {e}")

        # Final status embed
        final_embed = discord.Embed(
            title=f"{self.EMOJIS['download']} Sync Complete",
            description=(
                f"Network synchronization finished\n\n"
                f"Progress: `{'█' * 20}` 100%"
            ),
            color=(
                discord.Color.green() if success_count == total_cameras else 
                discord.Color.orange() if success_count > 0 else 
                discord.Color.red()
            ),
            timestamp=datetime.now()
        )

        final_embed.add_field(
            name="Results",
            value=(
                f"{self.EMOJIS['success']} Cameras Synced: `{success_count}/{total_cameras}`\n"
                f"{self.EMOJIS['download']} Files Downloaded: `{downloaded_files}`\n"
                f"{self.EMOJIS['storage']} Total Files Found: `{total_files}`"
            ),
            inline=False
        )

        final_embed.add_field(
            name="Storage Location",
            value=f"```\n{config.LOCAL_SAVE_PATH_1}\n```",
            inline=False
        )

        if success_count < total_cameras:
            final_embed.add_field(
                name=f"{self.EMOJIS['warning']} Sync Issues",
                value=(
                    "Some cameras failed to sync properly.\n"
                    "Check the system logs for detailed error information."
                ),
                inline=False
            )

        final_embed.set_footer(text=f"Network: {config.SMB_HOST} | Share: {config.SMB_SHARE}")
        await status_msg.edit(embed=final_embed)

    async def kill_command(self, interaction: discord.Interaction):
        """Gracefully shut down the bot and all services."""
        # Channel permission check
        if interaction.channel_id != config.COMMAND_CHANNEL_ID:
            error_embed = discord.Embed(
                title=f"{self.EMOJIS['error']} Unauthorized Channel",
                description="Please use the designated command channel for this operation.",
                color=discord.Color.red(),
                timestamp=datetime.now()
            ).set_footer(text="Access Denied")
            await interaction.response.send_message(embed=error_embed, ephemeral=True)
            return

        # Initial warning embed
        warning_embed = discord.Embed(
            title=f"{self.EMOJIS['warning']} System Shutdown",
            description="Initiating system shutdown sequence...",
            color=discord.Color.orange(),
            timestamp=datetime.now()
        )
        warning_embed.add_field(
            name="Confirmation Required",
            value=(
                "This will shut down all services and stop the bot.\n"
                "**Are you sure you want to continue?**\n\n"
                "✅ Confirm Shutdown\n"
                "❌ Cancel Operation"
            ),
            inline=False
        )
        warning_embed.add_field(
            name="Affected Services",
            value=(
                "• Motion Detection System\n"
                "• Camera Monitoring\n"
                "• Network Storage Connection\n"
                "• Discord Bot Interface"
            ),
            inline=False
        )
        warning_embed.set_footer(text="This action cannot be undone")
        
        msg = await interaction.response.send_message(embed=warning_embed)

        # Add reaction options
        message = await interaction.original_response()
        await message.add_reaction("✅")
        await message.add_reaction("❌")

        try:
            reaction, user = await self.wait_for(
                'reaction_add',
                timeout=30.0,
                check=lambda r, u: u == interaction.user and str(r.emoji) in ["✅", "❌"]
            )

            if str(reaction.emoji) == "✅":
                # Start shutdown sequence
                shutdown_embed = discord.Embed(
                    title=f"{self.EMOJIS['warning']} System Shutdown",
                    description="Executing shutdown sequence...",
                    color=discord.Color.red(),
                    timestamp=datetime.now()
                )
                shutdown_embed.add_field(
                    name="Progress",
                    value=f"{self.EMOJIS['motion']} Stopping services...",
                    inline=False
                )
                await message.edit(embed=shutdown_embed)

                # Stop cameras
                shutdown_embed.add_field(
                    name="Camera Systems",
                    value="Stopping camera monitoring...",
                    inline=False
                )
                await message.edit(embed=shutdown_embed)
                
                for camera in self.cameras:
                    camera.stop_monitoring()

                # Update status
                shutdown_embed.set_field_at(
                    1,
                    name="Camera Systems",
                    value=f"{self.EMOJIS['success']} Camera monitoring stopped",
                    inline=False
                )
                
                # Close network connections
                shutdown_embed.add_field(
                    name="Network Storage",
                    value="Closing network connections...",
                    inline=False
                )
                await message.edit(embed=shutdown_embed)
                
                if self.smb_uploader.conn:
                    self.smb_uploader.conn.close()
                
                shutdown_embed.set_field_at(
                    2,
                    name="Network Storage",
                    value=f"{self.EMOJIS['success']} Network connections closed",
                    inline=False
                )

                # Final status update
                shutdown_embed.set_field_at(
                    0,
                    name="Status",
                    value=(
                        f"{self.EMOJIS['success']} All services stopped\n"
                        f"{self.EMOJIS['warning']} System shutting down..."
                    ),
                    inline=False
                )
                await message.edit(embed=shutdown_embed)

                # Set shutdown flag and event
                self.shutting_down = True
                self.shutdown_event.set()
                
                # Final goodbye message
                final_embed = discord.Embed(
                    title=f"{self.EMOJIS['success']} Shutdown Complete",
                    description="All services have been stopped successfully.",
                    color=discord.Color.green(),
                    timestamp=datetime.now()
                )
                final_embed.add_field(
                    name="System Status",
                    value="System is now offline",
                    inline=False
                )
                await message.edit(embed=final_embed)
                
                # Close Discord connection
                await self.close()
                
            else:
                # Operation cancelled
                cancel_embed = discord.Embed(
                    title=f"{self.EMOJIS['error']} Operation Cancelled",
                    description="Shutdown sequence has been cancelled.",
                    color=discord.Color.blue(),
                    timestamp=datetime.now()
                )
                await message.edit(embed=cancel_embed)
                
        except asyncio.TimeoutError:
            # Timeout - no response
            timeout_embed = discord.Embed(
                title=f"{self.EMOJIS['error']} Operation Timed Out",
                description="No confirmation received. Shutdown cancelled.",
                color=discord.Color.red(),
                timestamp=datetime.now()
            )
            await message.edit(embed=timeout_embed)

    async def reboot_command(self, interaction: discord.Interaction):
        """Reboot the system with proper service shutdown."""
        # Channel permission check
        if interaction.channel_id != config.COMMAND_CHANNEL_ID:
            error_embed = discord.Embed(
                title=f"{self.EMOJIS['error']} Unauthorized Channel",
                description="Please use the designated command channel for this operation.",
                color=discord.Color.red(),
                timestamp=datetime.now()
            ).set_footer(text="Access Denied")
            await interaction.response.send_message(embed=error_embed, ephemeral=True)
            return

        # System check
        if platform.system() != 'Linux':
            error_embed = discord.Embed(
                title=f"{self.EMOJIS['error']} Unsupported Platform",
                description="The reboot command is only available on Linux systems.",
                color=discord.Color.red(),
                timestamp=datetime.now()
            )
            await interaction.response.send_message(embed=error_embed, ephemeral=True)
            return

        # Initial warning embed
        warning_embed = discord.Embed(
            title=f"{self.EMOJIS['warning']} System Reboot",
            description="System reboot sequence initiated.",
            color=discord.Color.orange(),
            timestamp=datetime.now()
        )
        warning_embed.add_field(
            name="Confirmation Required",
            value=(
                "This will reboot the entire system.\n"
                "**Are you sure you want to continue?**\n\n"
                "✅ Confirm Reboot\n"
                "❌ Cancel Operation"
            ),
            inline=False
        )
        warning_embed.set_footer(text="This action cannot be undone")
        
        msg = await interaction.response.send_message(embed=warning_embed)
        message = await interaction.original_response()
        await message.add_reaction("✅")
        await message.add_reaction("❌")

        try:
            reaction, user = await self.wait_for(
                'reaction_add',
                timeout=30.0,
                check=lambda r, u: u == interaction.user and str(r.emoji) in ["✅", "❌"]
            )

            if str(reaction.emoji) == "✅":
                # Create countdown embed
                countdown_embed = discord.Embed(
                    title=f"{self.EMOJIS['warning']} System Reboot",
                    description="Preparing for system restart...",
                    color=discord.Color.orange(),
                    timestamp=datetime.now()
                )
                await message.edit(embed=countdown_embed)

                # Stop cameras first
                for camera in self.cameras:
                    camera.stop_monitoring()

                # Countdown loop
                for i in range(10, 0, -1):
                    countdown_embed.clear_fields()
                    countdown_embed.add_field(
                        name="Status",
                        value=(
                            f"{self.EMOJIS['motion']} Stopping services...\n"
                            f"• Cameras stopped\n"
                            f"• Network connections closing\n"
                            f"• System preparing for reboot"
                        ),
                        inline=False
                    )
                    countdown_embed.add_field(
                        name="Time Remaining",
                        value=(
                            f"`{i}` seconds until reboot\n"
                            f"Progress: `{'█' * (10-i)}{'░' * i}` {((10-i)/10)*100:.0f}%"
                        ),
                        inline=False
                    )
                    await message.edit(embed=countdown_embed)
                    await asyncio.sleep(1)

                # Final reboot embed
                final_embed = discord.Embed(
                    title=f"{self.EMOJIS['warning']} System Reboot",
                    description="Executing system reboot...",
                    color=discord.Color.orange(),
                    timestamp=datetime.now()
                )
                final_embed.add_field(
                    name="Status",
                    value=(
                        f"{self.EMOJIS['success']} All services stopped\n"
                        f"{self.EMOJIS['motion']} Initiating reboot sequence\n"
                        "System will restart momentarily..."
                    ),
                    inline=False
                )
                await message.edit(embed=final_embed)

                # Set shutdown event and reboot
                self.shutdown_event.set()
                os.system('sudo reboot')
                
            else:
                # Operation cancelled
                cancel_embed = discord.Embed(
                    title=f"{self.EMOJIS['error']} Operation Cancelled",
                    description="Reboot sequence has been cancelled.",
                    color=discord.Color.blue(),
                    timestamp=datetime.now()
                )
                await message.edit(embed=cancel_embed)
                
        except asyncio.TimeoutError:
            # Timeout - no response
            timeout_embed = discord.Embed(
                title=f"{self.EMOJIS['error']} Operation Timed Out",
                description="No confirmation received. Reboot cancelled.",
                color=discord.Color.red(),
                timestamp=datetime.now()
            )
            await message.edit(embed=timeout_embed)
    async def status_command(self, interaction):
        """Comprehensive system status command with detailed metrics."""
        if interaction.channel_id != config.COMMAND_CHANNEL_ID:
            error_embed = discord.Embed(
                title=f"{self.EMOJIS['error']} Invalid Channel",
                description="This command can only be used in the designated channel.",
                color=discord.Color.red(),
                timestamp=datetime.now()
            )
            await interaction.response.send_message(embed=error_embed, ephemeral=True)
            return

        # Calculate uptime and stats
        uptime = datetime.now() - self.start_time
        uptime_str = str(uptime).split('.')[0]  # Remove microseconds

        status_embed = discord.Embed(
            title=f"{self.EMOJIS['status']} System Status Overview",
            description=(
                "Real-time system status and monitoring information\n\n"
                f"{self.EMOJIS['maintenance' if self.maintenance_mode else 'active']} "
                f"System Mode: `{'Maintenance' if self.maintenance_mode else 'Active'}`\n"
                f"⏱️ Uptime: `{uptime_str}`\n"
                f"🔄 Last Update: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
            ),
            color=discord.Color.orange() if self.maintenance_mode else discord.Color.blue(),
            timestamp=datetime.now()
        )

        # System State Field
        status_embed.add_field(
            name="🖥️ System State",
            value=(
                f"**Mode**: `{'Maintenance' if self.maintenance_mode else 'Active'}`\n"
                f"**Start Time**: `{self.start_time.strftime('%Y-%m-%d %H:%M:%S')}`\n"
                f"**Running Time**: `{uptime_str}`\n"
                f"**Status**: {self.EMOJIS['warning'] if self.maintenance_mode else self.EMOJIS['success']} "
                f"`{'Maintenance Mode' if self.maintenance_mode else 'Operational'}`"
            ),
            inline=False
        )

        # Camera Systems Status
        camera_status = []
        total_cameras = len(self.cameras)
        active_cameras = sum(1 for cam in self.cameras if cam.cap.isOpened())
        
        for camera in self.cameras:
            motion_emoji = "🟡" if camera.motion_value > config.MOTION_THRESHOLD else "⚪"
            
            # Get current camera settings
            resolution = (
                int(camera.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(camera.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            )
            fps = int(camera.cap.get(cv2.CAP_PROP_FPS))
            
            # Get latest frame for quality assessment
            ret, frame = camera.cap.read()
            quality_score = camera._calculate_quality_metrics(frame) if ret else 0
            
            # Quality indicator based on score
            quality_indicator = "🟢" if quality_score > 0.7 else "🟡" if quality_score > 0.4 else "🔴"
            
            status = (
                f"**Camera {camera.camera_index}**\n"
                f"{self.EMOJIS['connected'] if camera.cap.isOpened() else self.EMOJIS['disconnected']} "
                f"Status: {'`Online`' if camera.cap.isOpened() else '`Offline`'}\n"
                f"{motion_emoji} Motion Level: `{camera.motion_value:.2f}`\n"
                f"{'🔔' if camera.notifications_enabled else '🔕'} Notifications: "
                f"`{'Enabled' if camera.notifications_enabled else 'Disabled'}`\n"
                f"{'🔍' if camera.is_monitoring else '⏸️'} Monitoring: "
                f"`{'Active' if camera.is_monitoring else 'Paused'}`\n"
                f"📊 Quality Metrics:\n"
                f"├ Resolution: `{resolution[0]}x{resolution[1]}`\n"
                f"├ FPS: `{fps}`\n"
                f"└ Quality Score: {quality_indicator} `{quality_score:.2f}`"
            )
            camera_status.append(status)

        # Add Camera Systems field
        status_embed.add_field(
            name=f"{self.EMOJIS['camera']} Camera Systems",
            value=(
                f"**Overview**\n"
                f"Total Cameras: `{total_cameras}`\n"
                f"Active Cameras: `{active_cameras}`\n"
                f"System Health: {self.EMOJIS['success'] if active_cameras == total_cameras else self.EMOJIS['warning']}\n\n"
                f"**Individual Status**\n" + "\n\n".join(camera_status)
            ),
            inline=False
        )

        # Storage Systems Status
        try:
            # Get storage metrics
            total_primary = len(glob.glob(os.path.join(config.LOCAL_SAVE_PATH_1, "**/*.jpg"), recursive=True))
            total_backup = len(glob.glob(os.path.join(config.LOCAL_SAVE_PATH_2, "**/*.jpg"), recursive=True))
            
            # Calculate storage space
            primary_size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(config.LOCAL_SAVE_PATH_1, "**/*.jpg"), recursive=True))
            backup_size = sum(os.path.getsize(f) for f in glob.glob(os.path.join(config.LOCAL_SAVE_PATH_2, "**/*.jpg"), recursive=True))
            
            status_embed.add_field(
                name=f"{self.EMOJIS['storage']} Storage Systems",
                value=(
                    f"**Local Storage**\n"
                    f"Primary Path: `{config.LOCAL_SAVE_PATH_1}`\n"
                    f"├ Status: {self.EMOJIS['success'] if os.path.exists(config.LOCAL_SAVE_PATH_1) else self.EMOJIS['error']}\n"
                    f"├ Files: `{total_primary:,}`\n"
                    f"└ Size: `{primary_size/1024/1024:.1f} MB`\n\n"
                    f"Backup Path: `{config.LOCAL_SAVE_PATH_2}`\n"
                    f"├ Status: {self.EMOJIS['success'] if os.path.exists(config.LOCAL_SAVE_PATH_2) else self.EMOJIS['error']}\n"
                    f"├ Files: `{total_backup:,}`\n"
                    f"└ Size: `{backup_size/1024/1024:.1f} MB`"
                ),
                inline=False
            )
        except Exception as e:
            logging.error(f"Error getting storage metrics: {e}")
            status_embed.add_field(
                name=f"{self.EMOJIS['error']} Storage Error",
                value="Failed to retrieve storage metrics",
                inline=False
            )

        # Network Status
        connection_status = self.smb_uploader.check_connection()
        status_embed.add_field(
            name=f"{self.EMOJIS['connected'] if connection_status else self.EMOJIS['disconnected']} Network Storage",
            value=(
                f"**SMB Connection**\n"
                f"Status: `{'Connected' if connection_status else 'Disconnected'}`\n"
                f"Host: `{config.SMB_HOST}`\n"
                f"Share: `{config.SMB_SHARE}`\n"
                f"Last Connection: `{datetime.fromtimestamp(self.smb_uploader.last_connection_attempt).strftime('%H:%M:%S')}`"
            ),
            inline=False
        )

        # System Information
        system_info = platform.uname()
        status_embed.add_field(
            name="💻 System Information",
            value=(
                f"**System**: `{system_info.system}`\n"
                f"**Node**: `{system_info.node}`\n"
                f"**Release**: `{system_info.release}`\n"
                f"**Version**: `{system_info.version}`\n"
                f"**Machine**: `{system_info.machine}`\n"
                f"**Processor**: `{system_info.processor}`"
            ),
            inline=False
        )

        # Maintenance Mode Status (if active)
        if self.maintenance_mode:
            status_embed.add_field(
                name=f"{self.EMOJIS['maintenance']} Maintenance Status",
                value=(
                    "**Limited Functionality:**\n"
                    "• Camera monitoring paused\n"
                    "• Motion detection disabled\n"
                    "• Scheduled tasks suspended\n"
                    "• Image capture restricted\n\n"
                    f"{self.EMOJIS['warning']} Use `/maintenance` to exit maintenance mode"
                ),
                inline=False
            )

        # Add footer with refresh info
        status_embed.set_footer(
            text=(
                f"System Start: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')} | "
                f"Uptime: {uptime_str} | "
                "Use /status to refresh"
            )
        )

        await interaction.response.send_message(embed=status_embed)

    async def config_command(self, interaction):
        """Display comprehensive system configuration with formatted visual feedback."""
        # Channel validation
        if interaction.channel_id != config.COMMAND_CHANNEL_ID:
            error_embed = discord.Embed(
                title=f"{self.EMOJIS['error']} Invalid Channel",
                description="This command can only be used in the designated channel.",
                color=discord.Color.red(),
                timestamp=datetime.now()
            ).set_footer(text="Access Denied")
            await interaction.response.send_message(embed=error_embed, ephemeral=True)
            return

        # Create main configuration embed
        config_embed = discord.Embed(
            title=f"{self.EMOJIS['config']} System Configuration",
            description=(
                "Current system settings and configuration parameters\n\n"
                f"{self.EMOJIS['maintenance'] if self.maintenance_mode else self.EMOJIS['active']} "
                f"System Mode: `{'Maintenance' if self.maintenance_mode else 'Active'}`"
            ),
            color=discord.Color.orange() if self.maintenance_mode else discord.Color.blue(),
            timestamp=datetime.now()
        )

        # Camera Configuration Section
        camera_health = all(cam.cap.isOpened() for cam in self.cameras)
        active_cameras = sum(1 for cam in self.cameras if cam.cap.isOpened())
        camera_config = (
            f"**Configured Devices**\n"
            f"├ Available Cameras: `{', '.join(map(str, config.CAMERA_INDEXES))}`\n"
            f"└ Active Cameras: `{active_cameras}/{len(self.cameras)}`\n\n"
            f"**Motion Detection**\n"
            f"├ Detection Status: {self.EMOJIS['connected']} `Active`\n"
            f"├ Motion Threshold: `{config.MOTION_THRESHOLD}` units\n"
            f"├ Alert Cooldown: `{config.MOTION_ALERT_COOLDOWN}` minutes\n"
            f"└ System Health: {self.EMOJIS['success'] if camera_health else self.EMOJIS['warning']} "
            f"`{'Fully Operational' if camera_health else 'Degraded'}`"
        )
        config_embed.add_field(
            name=f"{self.EMOJIS['camera']} Camera Configuration",
            value=camera_config,
            inline=False
        )

        # Storage Configuration Section
        storage_config = (
            f"**Local Storage Paths**\n"
            f"├ Primary Storage\n"
            f"│  └ Path: `{config.LOCAL_SAVE_PATH_1}`\n"
            f"└ Backup Storage\n"
            f"   └ Path: `{config.LOCAL_SAVE_PATH_2}`\n\n"
            f"**Network Storage**\n"
            f"├ Connection Status: {self.EMOJIS['connected'] if self.smb_uploader.is_connected else self.EMOJIS['disconnected']}\n"
            f"├ Host: `{config.SMB_HOST}`\n"
            f"├ Share Name: `{config.SMB_SHARE}`\n"
            f"└ Remote Path: `{config.SMB_PATH}`"
        )
        config_embed.add_field(
            name=f"{self.EMOJIS['storage']} Storage Configuration",
            value=storage_config,
            inline=False
        )

        # Timing Configuration Section
        timing_config = (
            f"**Capture Parameters**\n"
            f"├ Minimum Interval: `{config.COOLDOWN_SECONDS}` seconds\n"
            f"└ Alert Cooldown: `{config.MOTION_ALERT_COOLDOWN * 60}` seconds\n\n"
            f"**Timelapse Settings**\n"
            f"└ Output FPS: `{config.FPS}` frames/second\n\n"
            f"**Scheduled Tasks**\n"
            f"├ {self.EMOJIS['camera']} Auto-Capture: `Hourly at :00`\n"
            f"└ {self.EMOJIS['upload']} Network Sync: `Daily at 00:00`"
        )
        config_embed.add_field(
            name="⏱️ Timing Configuration",
            value=timing_config,
            inline=False
        )

        # Discord Integration Section
        discord_config = (
            f"**Server Configuration**\n"
            f"├ Guild ID: `{config.GUILD_ID}`\n"
            f"└ Bot Status: {self.EMOJIS['connected']} `Online`\n\n"
            f"**Channel Configuration**\n"
            f"├ Commands Channel: <#{config.COMMAND_CHANNEL_ID}>\n"
            f"└ Motion Alerts: <#{config.MOTION_ALERT_CHANNEL_ID}>"
        )
        config_embed.add_field(
            name=f"{self.EMOJIS['config']} Discord Integration",
            value=discord_config,
            inline=False
        )

        # System State Section
        uptime = datetime.now() - self.start_time
        system_state = (
            f"**Runtime Information**\n"
            f"├ Start Time: `{self.start_time.strftime('%Y-%m-%d %H:%M:%S')}`\n"
            f"├ Uptime: `{str(uptime).split('.')[0]}`\n"
            f"└ Platform: `{platform.system()} {platform.release()}`\n\n"
            f"**Active Features**\n"
            f"├ Motion Detection: {'🟢' if not self.maintenance_mode else '🔴'}\n"
            f"├ Auto-Capture: {'🟢' if not self.maintenance_mode else '🔴'}\n"
            f"└ Network Sync: {'🟢' if self.smb_uploader.is_connected else '🔴'}"
        )
        config_embed.add_field(
            name="💻 System State",
            value=system_state,
            inline=False
        )

        # Add helpful tips in footer
        config_embed.set_footer(
            text=(
                f"Use /status for real-time metrics • "
                f"Settings loaded from cred.env • "
                f"Last Update: {datetime.now().strftime('%H:%M:%S')}"
            )
        )

        await interaction.response.send_message(embed=config_embed)

    async def maintenance_command(self, interaction: discord.Interaction):
        """Handle maintenance mode toggle and camera calibration."""
        
        # Channel validation
        if interaction.channel_id != config.COMMAND_CHANNEL_ID:
            await interaction.response.send_message(
                embed=discord.Embed(
                    title=f"{self.EMOJIS['error']} Invalid Channel",
                    description="This command can only be used in the designated channel.",
                    color=discord.Color.red(),
                    timestamp=datetime.now()
                ).set_footer(text="Access Denied"),
                ephemeral=True
            )
            return

        # Toggle maintenance mode
        entering = not self.maintenance_mode
        self.maintenance_mode = entering

        # Create initial embed
        status_embed = discord.Embed(
            title=f"{self.EMOJIS['maintenance']} Maintenance Mode",
            description=(
                f"{'Entering' if entering else 'Exiting'} maintenance mode...\n\n"
                f"System State: `{'Maintenance' if entering else 'Active'}`"
            ),
            color=discord.Color.orange() if entering else discord.Color.green(),
            timestamp=datetime.now()
        )

        if entering:
            # ENTERING MAINTENANCE MODE
            
            # Initial response
            status_embed.add_field(
                name="Status", 
                value=f"{self.EMOJIS['motion']} Stopping camera systems...",
                inline=False
            )
            await interaction.response.send_message(embed=status_embed)
            
            # Stop all cameras
            for camera in self.cameras:
                camera.stop_monitoring()
            
            # Update status and ask about calibration
            status_embed.clear_fields()
            status_embed.add_field(
                name="Camera Control",
                value=(
                    f"{self.EMOJIS['success']} Cameras stopped successfully\n\n"
                    "Would you like to calibrate cameras?\n"
                    "✅ Yes - Run calibration\n"
                    "❌ No - Skip calibration"
                ),
                inline=False
            )
            msg = await interaction.edit_original_response(embed=status_embed)
            
            # Add reaction options
            await msg.add_reaction("✅")
            await msg.add_reaction("❌")
            
            try:
                # Wait for user response
                reaction, user = await self.wait_for(
                    'reaction_add',
                    timeout=30.0,
                    check=lambda r, u: u == interaction.user and str(r.emoji) in ["✅", "❌"]
                )
                
                if str(reaction.emoji) == "✅":
                    # User chose to calibrate
                    for camera in self.cameras:
                        # Create calibration status embed
                        calib_embed = discord.Embed(
                            title=f"{self.EMOJIS['maintenance']} Camera Calibration",
                            description=f"Calibrating Camera {camera.camera_index}...",
                            color=discord.Color.blue(),
                            timestamp=datetime.now()
                        )
                        await msg.edit(embed=calib_embed)
                        
                        # Run calibration with progress updates
                        async for progress in camera.calibrate_camera():
                            calib_embed.clear_fields()
                            
                            # Calculate progress bars
                            res_progress = "█" * int(progress['resolution_progress'] * 20)
                            res_progress += "░" * (20 - len(res_progress))
                            
                            param_progress = "█" * int(progress['parameter_progress'] * 20)
                            param_progress += "░" * (20 - len(param_progress))
                            
                            # Add progress fields
                            calib_embed.add_field(
                                name="Resolution Testing",
                                value=(
                                    f"Current: `{progress['current_resolution']}`\n"
                                    f"Progress: `{res_progress}` "
                                    f"{progress['resolution_progress']*100:.0f}%"
                                ),
                                inline=False
                            )
                            
                            calib_embed.add_field(
                                name="Parameter Optimization",
                                value=(
                                    f"Testing: `{progress['current_parameter']}`\n"
                                    f"Progress: `{param_progress}` "
                                    f"{progress['parameter_progress']*100:.0f}%"
                                ),
                                inline=False
                            )
                            
                            if progress['best_score']:
                                calib_embed.add_field(
                                    name="Best Quality Score",
                                    value=f"`{progress['best_score']:.2f}`",
                                    inline=False
                                )
                            
                            await msg.edit(embed=calib_embed)
                            await asyncio.sleep(0.1)  # Prevent rate limiting
                            
            except asyncio.TimeoutError:
                # User didn't respond in time
                status_embed.clear_fields()
                status_embed.add_field(
                    name="Calibration",
                    value=f"{self.EMOJIS['warning']} Calibration skipped (timeout)",
                    inline=False
                )
                await msg.edit(embed=status_embed)
                
        else:
            # EXITING MAINTENANCE MODE
            
            # Initial response
            status_embed.add_field(
                name="Status",
                value=f"{self.EMOJIS['motion']} Restarting camera systems...",
                inline=False
            )
            await interaction.response.send_message(embed=status_embed)
            
            # Restart cameras
            success_count = 0
            for camera in self.cameras:
                if camera.start_monitoring():
                    success_count += 1
            
            # Update status
            status_embed.clear_fields()
            status_embed.add_field(
                name="Camera Status",
                value=(
                    f"{self.EMOJIS['success']} Restarted {success_count}/{len(self.cameras)} cameras\n"
                    f"System Mode: `Active`"
                ),
                inline=False
            )

        # Add final system state
        status_embed.add_field(
            name="System State",
            value=(
                f"Mode: `{'Maintenance' if entering else 'Active'}`\n"
                f"Cameras: `{'Stopped' if entering else 'Running'}`\n"
                f"Motion Detection: `{'Disabled' if entering else 'Enabled'}`\n"
                f"Scheduled Tasks: `{'Suspended' if entering else 'Active'}`"
            ),
            inline=False
        )
        
        status_embed.add_field(
            name="Available Commands",
            value=(
                f"{self.EMOJIS['config']} `/status` - View system status\n"
                f"{self.EMOJIS['camera']} `/capture` - Force image capture\n"
                f"{self.EMOJIS['maintenance']} `/maintenance` - Toggle mode"
            ),
            inline=False
        )
        
        status_embed.set_footer(
            text="Use /maintenance again to toggle maintenance mode"
        )
        
        # Final update
        if entering:
            await msg.edit(embed=status_embed)
        else:
            await interaction.edit_original_response(embed=status_embed)

def setup_scheduled_tasks(cameras, smb_uploader):
    def hourly_capture():
        if not bot.maintenance_mode and not bot.shutting_down:
            for camera in cameras:
                camera.capture_image(force=True)
            
    def midnight_upload():
        if not bot.maintenance_mode and not bot.shutting_down:
            smb_uploader.upload_pending()
    
    schedule.every().hour.at(":00").do(hourly_capture)
    schedule.every().day.at("00:00").do(midnight_upload)
    
    while not bot.shutdown_event.is_set():
        schedule.run_pending()
        time.sleep(1)

async def initialize_services():
    """Initialize all services required by the bot"""
    try:
        # Initialize SMB uploader first
        smb_uploader = SMBUploader()
        logging.info("SMB uploader initialized")
        
        # Initialize cameras with delay between each
        cameras = []
        for idx in config.CAMERA_INDEXES:
            try:
                # Add delay between camera initializations
                if len(cameras) > 0:
                    await asyncio.sleep(2)  # 2 second delay
                    
                # Try initialization with retries
                retries = 3
                camera = None
                
                for attempt in range(retries):
                    try:
                        camera = await Camera.create(idx, bot)
                        if camera and camera.cap.isOpened():
                            # Test camera with capture
                            ret, frame = camera.cap.read()
                            if ret and frame is not None:
                                break
                        # If not successful, try lower resolution
                        if camera:
                            camera.settings['resolution'] = (640, 480)
                            camera.apply_settings()
                    except Exception as e:
                        if attempt < retries - 1:
                            logging.warning(f"Retry {attempt + 1} for camera {idx}")
                            await asyncio.sleep(1)
                        else:
                            raise e
                            
                if camera and camera.cap.isOpened():
                    cameras.append(camera)
                    logging.info(f"Camera {idx} initialized and ready")
                else:
                    logging.warning(f"Camera {idx} failed to initialize properly")
                    
            except Exception as e:
                logging.error(f"Failed to initialize camera {idx}: {e}")
                continue
                
        if not cameras:
            logging.warning("No cameras were successfully initialized")
            
        return cameras, smb_uploader
        
    except Exception as e:
        logging.error(f"Service initialization failed: {e}")

def setup_signal_handlers(shutdown_func):
    """Set up signal handlers for graceful shutdown"""
    def handler(signum, _):
        logging.info(f"Received signal {signum}")
        asyncio.create_task(shutdown_func())
        
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, handler)

async def startup():
    """Main startup sequence"""
    global config, bot
    
    try:
        # Initialize bot with required intents
        intents = discord.Intents.default()
        intents.message_content = True
        bot = DiscordBot(intents=intents)
        
        # Set up signal handlers immediately
        setup_signal_handlers(bot.shutdown)
        
        # Connect bot first
        try:
            await bot.start(config.DISCORD_TOKEN)
        except Exception as e:
            logging.error(f"Failed to start bot: {e}")
            raise
            
    except Exception as e:
        logging.error(f"Startup failed: {e}")
        if 'bot' in globals() and hasattr(bot, 'shutdown'):
            await bot.shutdown()
        raise


async def cleanup():
    """Handle final cleanup operations"""
    try:
        if 'bot' in globals() and hasattr(bot, 'cameras'):
            for camera in bot.cameras:
                try:
                    camera.stop_monitoring()
                    if camera.cap and camera.cap.isOpened():
                        camera.cap.release()
                except Exception as e:
                    logging.error(f"Error stopping camera: {e}")
            logging.info("All cameras stopped")
        
        if 'bot' in globals() and hasattr(bot, 'smb_uploader'):
            if bot.smb_uploader and bot.smb_uploader.conn:
                try:
                    bot.smb_uploader.conn.close()
                    logging.info("SMB connection closed")
                except Exception as e:
                    logging.error(f"Error closing SMB connection: {e}")
                    
    except Exception as e:
        logging.error(f"Error during cleanup: {e}")

async def main():
    """Entry point with proper error handling"""
    try:
        # Initialize config first
        global config
        config = Config()
        
        # Start the bot
        await startup()
        
    except KeyboardInterrupt:
        logging.info("Received keyboard interrupt")
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
    finally:
        # Ensure cleanup runs
        await cleanup()


if __name__ == "__main__":
    try:
        # Set up logging first
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('timelapse.log'),
                logging.StreamHandler()
            ]
        )
        
        # Run the async main function
        asyncio.run(main())
        
    except KeyboardInterrupt:
        logging.info("Shutting down from keyboard interrupt")
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)