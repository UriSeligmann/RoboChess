import os
from ultralytics import YOLO
import cv2
import numpy as np

from typing import List, Tuple, Optional

from Debugger import Debugger
from Wrapper_Decorators import timeit_method, debug_entry_exit_method

# =============================================
#       YOLO MODEL & PREDICTION WRAPPER
# =============================================
class YOLOModel:
    """
    Manages loading a YOLO model for object detection and running predictions.
    """
    def __init__(self, model_path: str, debugger: Debugger) -> None:
        """
        Args:
            model_path (str): Path to the YOLO model weights (e.g., .pt file).
            debugger (Debugger): A Debugger instance for logging and image saving.
        """
        self.debugger = debugger
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}")
        self.model = YOLO(model_path)
        self.debugger.log("YOLO model loaded successfully.", level=0)

    @timeit_method(level=1)
    @debug_entry_exit_method(level=2)
    def predict(self, 
                image_path: str, 
                debug_image: Optional[np.ndarray] = None
               ) -> Tuple[List[Tuple[float, float]], np.ndarray]:
        """
        Runs YOLO prediction on the given image.

        Args:
            image_path (str): Path to the input image.
            debug_image (Optional[np.ndarray]): Optionally, a preloaded image 
                for drawing debug info.

        Returns:
            Tuple[List[Tuple[float, float]], np.ndarray]: 
                1) A list of centroid coordinates 
                2) The corresponding confidence scores for each detection.
        """
        results = self.model.predict(source=image_path, save=True)
        self.debugger.log("Prediction completed.", level=0)
        result = results[0]
        if not result.boxes:
            raise ValueError("No detections were made by the YOLO model.")

        boxes = result.boxes.xyxy.cpu().numpy()  # shape (N, 4) [x1, y1, x2, y2]
        confidences = result.boxes.conf.cpu().numpy()  # shape (N,)

        # Compute centroids
        centroids = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in boxes]

        # Optional debug visualization
        if self.debugger.debug_level >= 1:
            if debug_image is None:
                debug_image = cv2.imread(image_path)  # fallback read
            bboxes_img = self.debugger.draw_bboxes(debug_image, boxes, confidences)
            self.debugger.save_image(bboxes_img, "Raw_YOLO_Detections")

            centroids_img = self.debugger.draw_centroids(debug_image, centroids)
            self.debugger.save_image(centroids_img, "Centroids_Before_Filter")

        return centroids, confidences