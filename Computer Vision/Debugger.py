import os
import cv2
import numpy as np
from typing import List, Tuple, Optional

# =============================================
#               DEBUGGER CLASS
# =============================================
class Debugger:
    """
    Central debugging utility for logging messages, saving images, 
    and drawing bounding boxes/markers.
    """
    def __init__(self, debug_level: int = 0, save_directory: str = "Predictions") -> None:
        """
        Args:
            debug_level (int): Global debug verbosity level. Higher means more logs.
            save_directory (str): Directory to save debug images.
        """
        self.debug_level = debug_level
        self.save_directory = save_directory
        os.makedirs(save_directory, exist_ok=True)

    def log(self, message: str, level: int = 0) -> None:
        """
        Prints a debug message if the current debug level is >= message level.
        
        Args:
            message (str): The message to log.
            level (int): The importance/verbosity level for this message.
        """
        if self.debug_level >= level:
            print(f"[DEBUG] {message}")

    def save_image(self, image: np.ndarray, step_name: str) -> None:
        """
        Saves an image with a given step_name for debugging.
        
        Args:
            image (np.ndarray): The image to save.
            step_name (str): A label or identifier for this debug step.
        """
        filename = f"Debug_{step_name}.jpg"
        path = os.path.join(self.save_directory, filename)
        cv2.imwrite(path, image)
        self.log(f"Saved debug image at {path}", level=1)

    def draw_bboxes(self, 
                    image: np.ndarray, 
                    boxes: np.ndarray, 
                    confidences: np.ndarray) -> np.ndarray:
        """
        Draws bounding boxes with confidence scores on a copy of the image.
        
        Args:
            image (np.ndarray): The original image.
            boxes (np.ndarray): An array of shape (N, 4) with bounding boxes 
                [x1, y1, x2, y2].
            confidences (np.ndarray): An array of shape (N,) with confidence scores.

        Returns:
            np.ndarray: A copy of the original image with bounding boxes drawn.
        """
        out_img = image.copy()
        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(out_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out_img, f"{conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return out_img

    def draw_centroids(self, 
                       image: np.ndarray, 
                       centroids: List[Tuple[float, float]], 
                       color: Tuple[int, int, int] = (255, 0, 0), 
                       thickness: int = 3, 
                       radius: int = 5) -> np.ndarray:
        """
        Draws centroid circles on a copy of the image.

        Args:
            image (np.ndarray): The original image.
            centroids (List[Tuple[float, float]]): The list of (x, y) centroids.
            color (Tuple[int,int,int]): Circle color in BGR.
            thickness (int): Circle edge thickness.
            radius (int): Circle radius.

        Returns:
            np.ndarray: A copy of the image with centroid markers.
        """
        out_img = image.copy()
        for (cx, cy) in centroids:
            cv2.circle(out_img, (int(cx), int(cy)), radius, color, thickness)
        return out_img

    def draw_labeled_corners(self, 
                             image: np.ndarray, 
                             corners: List[Tuple[float, float]], 
                             labels: Optional[List[str]] = None) -> np.ndarray:
        """
        Draws corner points with labels on a copy of the image.

        Args:
            image (np.ndarray): The original image.
            corners (List[Tuple[float, float]]): Four corner coordinates.
            labels (List[str], optional): Labels for the corners. 
                Defaults to ["C1", "C2", "C3", "C4"].

        Returns:
            np.ndarray: A copy of the image with corners labeled.
        """
        if labels is None:
            labels = ["C1", "C2", "C3", "C4"]
        out_img = image.copy()
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255)]
        for i, point in enumerate(corners):
            x, y = map(int, point)
            color = colors[i % len(colors)]
            cv2.circle(out_img, (x, y), 8, color, -1)
            cv2.putText(out_img, labels[i], (x + 5, y - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        return out_img