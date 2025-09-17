import cv2
import os
import numpy as np
from debug_utils.DebugManager import DebugManager

from BoardDetector.yolo_model import YOLOModel
from BoardDetector.centroid_processor import CentroidProcessor
from BoardDetector.perspective_transformer import PerspectiveTransformer

# =============================================
#            MAIN ORCHESTRATOR
# =============================================
class ChessboardDetector:
    """
    High-level pipeline orchestrator for:
      - Loading the YOLO model and detecting corners.
      - Filtering/ordering centroids.
      - Applying perspective transform.
      - Drawing & warping back a visual grid overlay.
      - Extracting individual squares.
    """
    def __init__(self, 
                 image_path: str, 
                 model_path: str = "ChessBoardCornersPredictor.pt",
                 save_directory: str = "Predictions", 
                 centroid_min_distance: float = 50.0,
                 grid_divisions: int = 8, 
                 warp_chessboard_size: int = 800,
                 debug_level: int = 0) -> None:
        """
        Args:
            image_path (str): Path to the input chessboard image.
            model_path (str): Path to the YOLO model weights.
            save_directory (str): Directory to save all debug results and extracted squares.
            centroid_min_distance (float): Minimum distance used in centroid filtering.
            grid_divisions (int): Number of divisions (cells) in the grid (8 for standard chess).
            warp_chessboard_size (int): Pixel size of the output warped image.
            debug_level (int): Debug/verbosity level (0 = minimal logs, higher = more logs).
        """
        self.image_path = image_path
        self.original_image = cv2.imread(image_path)
        if self.original_image is None:
            raise ValueError(f"Failed to read image from {image_path}")

        self.save_directory = save_directory
        self.debug_level = debug_level
        os.makedirs(save_directory, exist_ok=True)

        # Use centralized debugger
        self.debugger = DebugManager(debug_level, save_directory)

        # Update sub-components to use same debugger instance
        self.yolo_model = YOLOModel(model_path)
        self.centroid_processor = CentroidProcessor(centroid_min_distance)
        self.transformer = PerspectiveTransformer(warp_chessboard_size)

        self.grid_divisions = grid_divisions

    def get_grid_overlay(self) -> np.ndarray:
        """
        Main pipeline method. Runs the entire detection → filtering → warp → 
        grid-drawing → square-extraction process.

        Returns:
            Dict[str, np.ndarray]: A dictionary mapping each grid cell label (e.g. 'A1') 
                to the corresponding square image.
        """

        # Run YOLO detection
        centroids, confidences = self.yolo_model.predict(
            self.image_path, 
            debug_image=self.original_image
        )

        # Filter and order centroids
        filtered = self.centroid_processor.filter_centroids(centroids, confidences)
        # If more than 4 remain, keep only top 4
        filtered = filtered[:4]
        ordered = self.centroid_processor.order_centroids(filtered)

        # Debug: draw ordered corners
        labeled_corners = self.debugger.draw_labeled_corners(
            self.original_image, ordered,
            labels=["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
        )
        self.debugger.save_image(labeled_corners, "Ordered_Corners")

        # Warp to a top-down chessboard
        matrix_path = os.path.join(self.save_directory, "transform_matrix.npy")
        warped, transform_matrix, src_corners, dst_corners = self.transformer.warp_image(self.original_image, ordered)
        self.transformer.save_transform_matrix(transform_matrix, matrix_path, src_corners=src_corners, dst_corners=dst_corners)
        
        self.debugger.save_image(warped, "Warped_ChessBoard")

        # Draw a precise grid on the warped image
        grid_warped = self.transformer.draw_precise_grid(warped, divisions=self.grid_divisions)
        self.debugger.save_image(grid_warped, "Warped_ChessBoard_Grid")

        return warped


# =============================================
#                MAIN EXECUTION
# =============================================
if __name__ == "__main__":

    detector = ChessboardDetector(
        image_path=r"",
        model_path="Models/ChessBoardCornersPredictor.pt",
        save_directory="Board_Detector_Debug",
        centroid_min_distance=0.1,
        grid_divisions=8,
        warp_chessboard_size=800,
        debug_level=2
    )
    warped = detector.get_grid_overlay()