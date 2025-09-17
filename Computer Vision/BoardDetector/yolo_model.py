import os
from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Tuple, Optional

# =============================================
#       YOLO MODEL & PREDICTION WRAPPER
# =============================================
class YOLOModel:
    """
    Manages loading a YOLO model for object detection and running predictions.
    """
    def __init__(self, model_path: str) -> None:
        """
        Args:
            model_path (str): Path to the YOLO model weights (e.g., .pt file).
        """
        self.model_path = model_path
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}")
        self.model = YOLO(model_path)

    @staticmethod
    def enhance_contrast(img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

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
        img = cv2.imread(image_path)
        img = self.enhance_contrast(img)
        cv2.imwrite("_tmp_resized.jpg", img)
        results = self.model.predict(source="_tmp_resized.jpg", conf=0.01)
        os.remove("_tmp_resized.jpg")
        result = results[0]
        if not result.boxes:
            raise ValueError("No detections were made by the YOLO model.")

        boxes = result.boxes.xyxy.cpu().numpy()  # shape (N, 4) [x1, y1, x2, y2]
        confidences = result.boxes.conf.cpu().numpy()  # shape (N,)

        # Compute centroids
        centroids = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in boxes]

        min_distance = 50.0  # pixels; tune to your resolution

        filtered_centroids = []
        filtered_confidences = []
        filtered_boxes = []

        # sort detections by confidence descending
        idx = np.argsort(-confidences)
        sorted_boxes = boxes[idx]
        sorted_conf = confidences[idx]
        sorted_centroids = [centroids[i] for i in idx]

        for box, c, conf in zip(sorted_boxes, sorted_centroids, sorted_conf):
            if all(np.linalg.norm(np.subtract(c, p)) > min_distance for p in filtered_centroids):
                filtered_centroids.append(c)
                filtered_confidences.append(conf)
                filtered_boxes.append(box)

        centroids = filtered_centroids
        confidences = np.array(filtered_confidences)
        boxes = np.array(filtered_boxes)


        return centroids, confidences