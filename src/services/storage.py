import os
import logging
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
from datetime import datetime
from smb.SMBConnection import SMBConnection

from ..config.settings import StorageSettings
from ..utils.logging import setup_logger

logger = setup_logger(__name__)

class StorageManager:
    """Manages local and network storage operations for camera images"""
    
    def __init__(self, settings: StorageSettings):
        self.settings = settings
        self.smb_conn = None
        self.connected = False
        
        # Ensure storage directories exist
        self._initialize_storage()
        
    def _initialize_storage(self) -> None:
        """Initialize local storage directories"""
        try:
            # Create primary and backup directories if needed
            os.makedirs(self.settings.primary_path, exist_ok=True) 
            os.makedirs(self.settings.backup_path, exist_ok=True)
            
            # Create camera subdirectories
            for path in [self.settings.primary_path, self.settings.backup_path]:
                for camera_idx in range(10):  # Support up to 10 cameras
                    camera_dir = self.settings.get_camera_path(path, camera_idx)
                    os.makedirs(camera_dir, exist_ok=True)
                    
        except Exception as e:
            logger.error(f"Storage initialization failed: {e}")
            raise
            
    def connect_network(self) -> bool:
        """Establish network storage connection"""
        try:
            if self.smb_conn:
                self.smb_conn.close()
                
            self.smb_conn = SMBConnection(
                self.settings.smb_username,
                self.settings.smb_password,
                "TimeLapseClient",
                self.settings.smb_host,
                use_ntlm_v2=True
            )
            
            self.connected = self.smb_conn.connect(
                self.settings.smb_host, 
                445, 
                timeout=30
            )
            
            if self.connected:
                # Verify share access
                self.smb_conn.listPath(self.settings.smb_share, '/')
                logger.info("Network storage connected")
            
            return self.connected
            
        except Exception as e:
            logger.error(f"Network connection failed: {e}")
            self.connected = False
            return False
            
    def disconnect_network(self) -> None:
        """Close network storage connection"""
        if self.smb_conn:
            try:
                self.smb_conn.close()
            except:
                pass
            finally:
                self.smb_conn = None
                self.connected = False
                
    def save_image(
        self,
        image_data: bytes,
        camera_index: int,
        timestamp: Optional[datetime] = None
    ) -> Tuple[str, str]:
        """
        Save image to primary and backup storage
        
        Returns:
            Tuple of (primary_path, backup_path)
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        filename = f"camera_{camera_index}_{timestamp.strftime('%Y%m%d_%H%M%S')}.jpg"
        
        # Get storage paths
        primary_path = os.path.join(
            self.settings.get_camera_path(self.settings.primary_path, camera_index),
            filename
        )
        backup_path = os.path.join(
            self.settings.get_camera_path(self.settings.backup_path, camera_index),
            filename
        )
        
        try:
            # Save to both locations
            for path in [primary_path, backup_path]:
                with open(path, 'wb') as f:
                    f.write(image_data)
                    
            logger.debug(f"Saved image {filename} to storage")
            return primary_path, backup_path
            
        except Exception as e:
            logger.error(f"Failed to save image {filename}: {e}")
            raise

    def upload_pending(self, camera_index: Optional[int] = None) -> bool:
        """
        Upload pending images to network storage
        
        Args:
            camera_index: Optional specific camera to upload
            
        Returns:
            Success status
        """
        if not self.connected and not self.connect_network():
            return False
            
        success = True
        try:
            # Get camera directories to process
            if camera_index is not None:
                cameras = [camera_index]
            else:
                # Get all camera directories that exist
                cameras = []
                for d in os.listdir(self.settings.primary_path):
                    if d.startswith("camera") and os.path.isdir(
                        os.path.join(self.settings.primary_path, d)
                    ):
                        try:
                            cam_idx = int(d.split("camera")[-1])
                            cameras.append(cam_idx)
                        except:
                            continue
                            
            # Process each camera directory
            for cam_idx in cameras:
                local_path = self.settings.get_camera_path(
                    self.settings.primary_path, 
                    cam_idx
                )
                remote_path = f"camera{cam_idx}/"
                
                try:
                    # Ensure remote directory exists
                    try:
                        self.smb_conn.listPath(self.settings.smb_share, remote_path)
                    except:
                        self.smb_conn.createDirectory(
                            self.settings.smb_share, 
                            remote_path
                        )
                        
                    # Upload each image file
                    for filename in os.listdir(local_path):
                        if not filename.endswith('.jpg'):
                            continue
                            
                        local_file = os.path.join(local_path, filename)
                        remote_file = f"{remote_path}{filename}"
                        
                        with open(local_file, 'rb') as f:
                            self.smb_conn.storeFile(
                                self.settings.smb_share,
                                remote_file,
                                f
                            )
                            logger.debug(f"Uploaded {filename}")
                            
                except Exception as e:
                    logger.error(f"Error uploading camera {cam_idx}: {e}")
                    success = False
                    
            return success
            
        except Exception as e:
            logger.error(f"Upload operation failed: {e}")
            return False
            
    def sync_from_network(self, camera_index: Optional[int] = None) -> bool:
        """
        Sync images from network storage to local storage
        
        Args:
            camera_index: Optional specific camera to sync
            
        Returns:
            Success status
        """
        if not self.connected and not self.connect_network():
            return False
            
        success = True
        try:
            # Determine cameras to sync
            if camera_index is not None:
                cameras = [camera_index]
            else:
                # List all camera directories on network
                cameras = []
                network_dirs = self.smb_conn.listPath(self.settings.smb_share, '/')
                for d in network_dirs:
                    if d.filename.startswith("camera"):
                        try:
                            cam_idx = int(d.filename.split("camera")[-1])
                            cameras.append(cam_idx)
                        except:
                            continue
                            
            # Process each camera
            for cam_idx in cameras:
                remote_path = f"camera{cam_idx}/"
                local_path = self.settings.get_camera_path(
                    self.settings.primary_path,
                    cam_idx
                )
                
                try:
                    # Ensure local directory exists
                    os.makedirs(local_path, exist_ok=True)
                    
                    # List remote files
                    remote_files = self.smb_conn.listPath(
                        self.settings.smb_share,
                        remote_path
                    )
                    
                    # Download missing files
                    for file_info in remote_files:
                        if not file_info.filename.endswith('.jpg'):
                            continue
                            
                        local_file = os.path.join(local_path, file_info.filename)
                        if not os.path.exists(local_file):
                            remote_file = f"{remote_path}{file_info.filename}"
                            
                            with open(local_file, 'wb') as f:
                                self.smb_conn.retrieveFile(
                                    self.settings.smb_share,
                                    remote_file,
                                    f
                                )
                                logger.debug(f"Downloaded {file_info.filename}")
                                
                            # Copy to backup location
                            backup_file = os.path.join(
                                self.settings.get_camera_path(
                                    self.settings.backup_path,
                                    cam_idx
                                ),
                                file_info.filename
                            )
                            shutil.copy2(local_file, backup_file)
                            
                except Exception as e:
                    logger.error(f"Error syncing camera {cam_idx}: {e}")
                    success = False
                    
            return success
            
        except Exception as e:
            logger.error(f"Sync operation failed: {e}")
            return False
            
    def get_storage_stats(self) -> dict:
        """Get storage statistics"""
        stats = {
            'primary': {'total': 0, 'cameras': {}},
            'backup': {'total': 0, 'cameras': {}},
            'network': {
                'connected': self.connected,
                'total': 0,
                'cameras': {}
            }
        }
        
        # Local storage stats
        for storage_type in ['primary', 'backup']:
            base_path = (
                self.settings.primary_path if storage_type == 'primary'
                else self.settings.backup_path
            )
            
            for cam_idx in range(10):
                camera_path = self.settings.get_camera_path(base_path, cam_idx)
                if os.path.exists(camera_path):
                    files = [f for f in os.listdir(camera_path) if f.endswith('.jpg')]
                    stats[storage_type]['cameras'][cam_idx] = len(files)
                    stats[storage_type]['total'] += len(files)
                    
        # Network storage stats if connected
        if self.connected:
            try:
                for cam_idx in range(10):
                    remote_path = f"camera{cam_idx}/"
                    try:
                        files = self.smb_conn.listPath(
                            self.settings.smb_share,
                            remote_path
                        )
                        jpg_count = len([f for f in files if f.filename.endswith('.jpg')])
                        stats['network']['cameras'][cam_idx] = jpg_count
                        stats['network']['total'] += jpg_count
                    except:
                        continue
            except:
                pass
                
        return stats