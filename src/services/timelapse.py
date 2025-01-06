import os
import cv2
import logging
import glob
from typing import Tuple, Optional
from datetime import datetime
from pathlib import Path

from ..config.settings import StorageSettings
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class TimelapseGenerator:
    """Handles timelapse video generation from captured images"""
    
    def __init__(self, settings: StorageSettings):
        self.settings = settings
        self.output_dir = "timelapses"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def create_timelapse(
        self,
        camera_index: int,
        fps: int = 30,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Generate a timelapse video from captured images
        
        Args:
            camera_index: Camera number
            fps: Output video FPS
            start_time: Optional start timestamp filter
            end_time: Optional end timestamp filter
            
        Returns:
            Tuple of (output_path, error_message)
        """
        try:
            # Get list of images for camera
            camera_dir = self.settings.get_camera_path(
                self.settings.primary_path,
                camera_index
            )
            
            if not os.path.exists(camera_dir):
                return None, f"No images found for camera {camera_index}"
                
            # Get all jpg files
            images = sorted(glob.glob(os.path.join(camera_dir, "*.jpg")))
            if not images:
                return None, "No images found for timelapse"
                
            # Filter by time range if specified
            if start_time or end_time:
                filtered_images = []
                for img_path in images:
                    timestamp = self._parse_timestamp(img_path)
                    if timestamp:
                        if start_time and timestamp < start_time:
                            continue
                        if end_time and timestamp > end_time:
                            continue
                    filtered_images.append(img_path)
                images = filtered_images
                
            if not images:
                return None, "No images found in specified time range"
                
            # Get output dimensions from first image
            first_frame = cv2.imread(images[0])
            height, width = first_frame.shape[:2]
            
            # Create output video
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(
                self.output_dir,
                f"timelapse_camera{camera_index}_{timestamp}.mp4"
            )
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(
                output_path,
                fourcc,
                fps,
                (width, height)
            )
            
            # Add each frame
            for img_path in images:
                frame = cv2.imread(img_path)
                if frame is not None:
                    out.write(frame)
                    
            out.release()
            logger.info(f"Created timelapse: {output_path}")
            
            return output_path, None
            
        except Exception as e:
            logger.error(f"Error creating timelapse: {e}")
            return None, str(e)
            
    def _parse_timestamp(self, filepath: str) -> Optional[datetime]:
        """Extract timestamp from image filename"""
        try:
            # Filename format: camera_X_YYYYMMDD_HHMMSS.jpg
            filename = os.path.basename(filepath)
            date_str = filename.split('_')[2:4]  # Get date and time parts
            timestamp = datetime.strptime('_'.join(date_str), '%Y%m%d_%H%M%S')
            return timestamp
        except:
            return None