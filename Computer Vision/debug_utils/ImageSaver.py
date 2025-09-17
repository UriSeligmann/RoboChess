import os
import cv2
import numpy as np
from pathlib import Path
from typing import Optional

class ImageSaver:
    """Handles saving debug images with proper directory management."""
    
    def __init__(self, base_directory: str = "debug_images"):
        self.base_directory = Path(base_directory)
        self.base_directory.mkdir(exist_ok=True, parents=True)
        
    def save(self, image: np.ndarray, name: str, subdirectory: Optional[str] = None) -> str:
        """Save an image to the debug directory."""
        if subdirectory:
            save_dir = self.base_directory / subdirectory
            save_dir.mkdir(exist_ok=True, parents=True)
        else:
            save_dir = self.base_directory
            
        filename = f"Debug_{name}.jpg"
        save_path = save_dir / filename
        cv2.imwrite(str(save_path), image)
        return str(save_path)
