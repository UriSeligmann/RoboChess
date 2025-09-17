import cv2
import numpy as np
from typing import List, Tuple, Optional

class ImagePainter:
    """Utility class for drawing various elements on images."""

    @staticmethod
    def draw_bboxes(image: np.ndarray, 
                    boxes: np.ndarray, 
                    confidences: np.ndarray) -> np.ndarray:
        """Draw bounding boxes with confidence scores."""
        out_img = image.copy()
        for box, conf in zip(boxes, confidences):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(out_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(out_img, f"{conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return out_img

    @staticmethod
    def draw_centroids(image: np.ndarray, 
                      centroids: List[Tuple[float, float]], 
                      color: Tuple[int, int, int] = (255, 0, 0), 
                      thickness: int = 3, 
                      radius: int = 5) -> np.ndarray:
        """Draw centroid circles."""
        out_img = image.copy()
        for (cx, cy) in centroids:
            cv2.circle(out_img, (int(cx), int(cy)), radius, color, thickness)
        return out_img

    @staticmethod
    def draw_labeled_corners(image: np.ndarray, 
                           corners: List[Tuple[float, float]], 
                           labels: Optional[List[str]] = None) -> np.ndarray:
        """Draw corner points with labels."""
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

    @staticmethod
    def draw_grid(image: np.ndarray, cell_size: Tuple[int, int], 
                  color: Tuple[int, int, int] = (128, 128, 128), 
                  thickness: int = 1) -> np.ndarray:
        """Draw a grid on the image."""
        h, w = image.shape[:2]
        cell_h, cell_w = cell_size
        out_img = image.copy()
        
        # Draw vertical lines
        for i in range(9):
            cv2.line(out_img, (i * cell_w, 0), (i * cell_w, h), color, thickness)
        
        # Draw horizontal lines
        for i in range(9):
            cv2.line(out_img, (0, i * cell_h), (w, i * cell_h), color, thickness)
            
        return out_img

    @staticmethod
    def draw_piece_annotations(image: np.ndarray, 
                             position: Tuple[int, int], 
                             piece_type: str,
                             cell_size: Tuple[int, int]) -> np.ndarray:
        """Draw piece circle and label at specified position."""
        out_img = image.copy()
        r, c = position
        cell_h, cell_w = cell_size
        
        # Calculate center position
        centre = (c * cell_w + cell_w // 2, r * cell_h + cell_h // 2)
        radius = cell_w // 4
        
        if piece_type == 'W':
            cv2.circle(out_img, centre, radius, (255, 255, 255), 2)
            cv2.putText(out_img, 'W', (centre[0]-10, centre[1]+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        elif piece_type == 'B':
            cv2.circle(out_img, centre, radius, (0, 0, 0), 2)
            cv2.putText(out_img, 'B', (centre[0]-10, centre[1]+5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
        return out_img

    @staticmethod
    def add_coordinate_label(image: np.ndarray, 
                           position: Tuple[int, int], 
                           cell_size: Tuple[int, int]) -> np.ndarray:
        """Add chess coordinate label to a cell."""
        out_img = image.copy()
        r, c = position
        cell_h, cell_w = cell_size
        
        coord_text = f"{chr(ord('a')+c)}{8-r}"
        cv2.putText(out_img, coord_text, 
                   (c * cell_w + 2, r * cell_h + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
        
        return out_img
