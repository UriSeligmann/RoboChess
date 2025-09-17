from typing import Optional
import numpy as np
from pathlib import Path

from .LoggingManager import LoggingManager
from .ImageSaver import ImageSaver
from .ImagePainter import ImagePainter

class DebugManager:
    """Central debug coordination for logging, image saving, and visualization."""
    
    def __init__(self, debug_level: int = 0, save_directory: str = "debug_output"):
        self.debug_level = debug_level
        self.save_directory = Path(save_directory)
        self.save_directory.mkdir(exist_ok=True, parents=True)
        
        # Initialize utilities
        self.logger = LoggingManager("CVDebugger", level=debug_level)
        self.image_saver = ImageSaver(save_directory)
        self.painter = ImagePainter()
    
    def log(self, message: str, level: int = 0) -> None:
        """Log message if current debug level is sufficient."""
        if self.debug_level >= level:
            if level >= 2:
                self.logger.debug(message)
            else:
                self.logger.info(message)
    
    def save_image(self, image: np.ndarray, name: str, 
                  subdirectory: Optional[str] = None) -> None:
        """Save debug image with optional subdirectory."""
        path = self.image_saver.save(image, name, subdirectory)
        self.log(f"Saved debug image: {path}", level=1)

    # Forward all drawing operations to ImagePainter
    def __getattr__(self, name):
        """Delegate drawing methods to ImagePainter."""
        if hasattr(self.painter, name):
            return getattr(self.painter, name)
        raise AttributeError(f"'DebugManager' has no attribute '{name}'")
    def draw_labeled_corners(self, *args, **kwargs):
        return self.painter.draw_labeled_corners(*args, **kwargs)
        
    def draw_grid(self, *args, **kwargs):
        return self.painter.draw_grid(*args, **kwargs)
        
    def draw_piece_annotations(self, *args, **kwargs):
        return self.painter.draw_piece_annotations(*args, **kwargs)
        
    def add_coordinate_label(self, *args, **kwargs):
        return self.painter.add_coordinate_label(*args, **kwargs)
