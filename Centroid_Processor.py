import numpy as np

from typing import List, Tuple, Optional

from Debugger import Debugger
from Wrapper_Decorators import debug_entry_exit_method


# =============================================
#      CENTROID FILTERING & ORDERING
# =============================================
class CentroidProcessor:
    """
    Filters and orders centroids (e.g. corners) from detection results.
    """
    def __init__(self, 
                 min_distance: float = 50.0, 
                 debugger: Optional[Debugger] = None) -> None:
        """
        Args:
            min_distance (float): Minimum distance threshold to treat centroids as distinct.
            debugger (Debugger, optional): Debugger instance for logging.
        """
        self.min_distance = min_distance
        self.debugger = debugger

    @debug_entry_exit_method(level=2)
    def filter_centroids(self, 
                         centroids: List[Tuple[float, float]], 
                         confidences: np.ndarray
                        ) -> List[Tuple[float, float]]:
        """
        Filters centroid detections based on a minimum distance threshold and 
        sorts by descending confidence.

        Args:
            centroids (List[Tuple[float, float]]): Centroid coordinates.
            confidences (np.ndarray): Confidence scores.

        Returns:
            List[Tuple[float, float]]: A filtered list of centroids.
        """
        if not centroids:
            return []

        # Sort by descending confidence
        sorted_indices = np.argsort(-confidences)
        centroids_sorted = [centroids[i] for i in sorted_indices]

        filtered = []
        for c in centroids_sorted:
            if all(np.linalg.norm(np.array(c) - np.array(existing)) > self.min_distance
                   for existing in filtered):
                filtered.append(c)
            if len(filtered) == 4:
                break

        if self.debugger:
            self.debugger.log(f"Detections after filtering: {len(filtered)}", level=0)

        return filtered

    @debug_entry_exit_method(level=2)
    def order_centroids(self, centroids: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Orders exactly 4 centroids in the order: [Top-Left, Top-Right, Bottom-Right, Bottom-Left].

        Args:
            centroids (List[Tuple[float, float]]): 4 distinct centroids.

        Raises:
            ValueError: If the number of centroids != 4.

        Returns:
            List[Tuple[float, float]]: Ordered corners (TL, TR, BR, BL).
        """
        if len(centroids) != 4:
            raise ValueError("Exactly 4 centroids are required to order them.")

        pts = np.array(centroids, dtype="float32")
        # Sort by y-coordinate (ascending)
        pts_sorted = pts[np.argsort(pts[:, 1])]
        top_row, bottom_row = pts_sorted[:2], pts_sorted[2:]
        # Sort left-to-right within each row
        top_left, top_right = top_row[np.argsort(top_row[:, 0])]
        bottom_left, bottom_right = bottom_row[np.argsort(bottom_row[:, 0])]

        return [tuple(top_left), tuple(top_right), tuple(bottom_right), tuple(bottom_left)]
