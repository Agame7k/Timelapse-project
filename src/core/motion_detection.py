import cv2
import numpy as np
import logging
from typing import Tuple, List, Optional
from dataclasses import dataclass

from ..config.settings import MotionConfig
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

@dataclass
class MotionEvent:
    """Data class to hold motion detection results"""
    detected: bool
    motion_value: float
    frame: Optional[np.ndarray]
    contours: List[np.ndarray]
    timestamp: float
    consecutive_frames: int

class MotionDetector:
    """Handles motion detection with configurable parameters"""
    
    def __init__(self, config: MotionConfig):
        self.config = config
        self.last_frame = None
        self.motion_buffer = []
        self.consecutive_motion = 0
        self.motion_value = 0.0
        
        # Motion detection parameters
        self.buffer_size = config.buffer_size
        self.min_motion_frames = config.min_motion_frames
        self.threshold = config.motion_threshold
        self.min_area = config.min_motion_area
        self.blur_size = config.blur_size
        self.dilate_iterations = config.dilate_iterations
        
    def detect(self, frame: np.ndarray) -> MotionEvent:
        """
        Detect motion in frame using configurable parameters and multi-stage validation.
        
        Args:
            frame: Current video frame
            
        Returns:
            MotionEvent containing detection results
        """
        if frame is None:
            return MotionEvent(False, 0.0, None, [], 0.0, 0)

        try:
            # Convert to grayscale and reduce noise
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)
            
            # Initialize reference frame if needed
            if self.last_frame is None:
                self.last_frame = blurred
                return MotionEvent(False, 0.0, frame, [], 0.0, 0)

            # Calculate frame difference
            frame_delta = cv2.absdiff(self.last_frame, blurred)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            
            # Apply morphological operations
            thresh = cv2.dilate(thresh, None, iterations=self.dilate_iterations)
            contours, _ = cv2.findContours(
                thresh.copy(), 
                cv2.RETR_EXTERNAL, 
                cv2.CHAIN_APPROX_SIMPLE
            )

            # Filter significant contours
            frame_area = frame.shape[0] * frame.shape[1]
            significant_contours = []
            
            for contour in contours:
                if cv2.contourArea(contour) > (frame_area * self.min_area):
                    significant_contours.append(contour)

            # Calculate motion percentage
            motion_pixels = np.sum(thresh) / 255.0
            self.motion_value = (motion_pixels / frame_area) * 100

            # Update motion buffer
            self.motion_buffer.append(self.motion_value)
            if len(self.motion_buffer) > self.buffer_size:
                self.motion_buffer.pop(0)

            # Check for sustained motion
            motion_detected = False
            if len(self.motion_buffer) == self.buffer_size:
                avg_motion = sum(self.motion_buffer) / self.buffer_size
                if avg_motion > self.threshold:
                    self.consecutive_motion += 1
                    if self.consecutive_motion >= self.min_motion_frames:
                        motion_detected = True
                else:
                    self.consecutive_motion = 0

            # Update reference frame
            self.last_frame = blurred
            
            return MotionEvent(
                detected=motion_detected,
                motion_value=self.motion_value,
                frame=frame,
                contours=significant_contours,
                timestamp=0.0,  # Set by caller
                consecutive_frames=self.consecutive_motion
            )

        except Exception as e:
            logger.error(f"Motion detection error: {e}")
            return MotionEvent(False, 0.0, None, [], 0.0, 0)

    def draw_motion_boxes(
        self, 
        frame: np.ndarray,
        contours: List[np.ndarray],
        show_metrics: bool = True
    ) -> np.ndarray:
        """
        Draw motion detection visualization on frame.
        
        Args:
            frame: Frame to draw on
            contours: List of detected contours
            show_metrics: Whether to show motion metrics
            
        Returns:
            Frame with motion boxes and visualization
        """
        frame_with_boxes = frame.copy()
        frame_height, frame_width = frame.shape[:2]
        
        if not contours:
            return frame_with_boxes

        # Process each detected object
        for idx, contour in enumerate(contours):
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate moments for centroid
            moments = cv2.moments(contour)
            if moments["m00"] == 0:
                continue
                
            # Calculate key points
            center_x = int(moments["m10"] / moments["m00"])
            center_y = int(moments["m01"] / moments["m00"])
            
            # Draw bounding box
            cv2.rectangle(
                frame_with_boxes,
                (x, y), 
                (x + w, y + h),
                (0, 255, 0),
                2
            )
            
            if show_metrics:
                # Add motion metrics
                cv2.putText(
                    frame_with_boxes,
                    f"Motion: {self.motion_value:.1f}%",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2
                )
                
                # Add object size
                size_text = f"{w}x{h}px"
                cv2.putText(
                    frame_with_boxes,
                    size_text,
                    (x, y + h + 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )

        return frame_with_boxes

    def reset(self):
        """Reset detector state"""
        self.last_frame = None
        self.motion_buffer.clear()
        self.consecutive_motion = 0
        self.motion_value = 0.0