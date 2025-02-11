import cv2
import os
import numpy as np
from typing import Dict

from Wrapper_Decorators import timeit_method, debug_entry_exit_method

from Debugger import Debugger
from YOLOModel import YOLOModel
from Centroid_Processor import CentroidProcessor
from PerspectiveTransformer import PerspectiveTransformer
from ChessboardSquareExtractor import ChessboardSquareExtractor

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

        # Shared debugger instance
        self.debugger = Debugger(debug_level, save_directory)

        # Sub-component initialization
        self.yolo_model = YOLOModel(model_path, self.debugger)
        self.centroid_processor = CentroidProcessor(centroid_min_distance, self.debugger)
        self.transformer = PerspectiveTransformer(warp_chessboard_size, self.debugger)
        self.square_extractor = ChessboardSquareExtractor(save_directory, grid_divisions)
        self.grid_divisions = grid_divisions

    @timeit_method(level=1)
    @debug_entry_exit_method(level=2)
    def cut_squares(self) -> Dict[str, np.ndarray]:
        """
        Main pipeline method. Runs the entire detection → filtering → warp → 
        grid-drawing → square-extraction process.

        Returns:
            Dict[str, np.ndarray]: A dictionary mapping each grid cell label (e.g. 'A1') 
                to the corresponding square image.
        """
        # 1. Run YOLO detection
        centroids, confidences = self.yolo_model.predict(
            self.image_path, 
            debug_image=self.original_image
        )

        # 2. Filter and order centroids
        filtered = self.centroid_processor.filter_centroids(centroids, confidences)
        if len(filtered) < 4:
            raise ValueError("Not enough centroids for perspective transformation.")
        # If more than 4 remain, keep only top 4
        filtered = filtered[:4]
        ordered = self.centroid_processor.order_centroids(filtered)

        # 3. Debug: draw ordered corners
        labeled_corners = self.debugger.draw_labeled_corners(
            self.original_image, ordered,
            labels=["Top-Left", "Top-Right", "Bottom-Right", "Bottom-Left"]
        )
        self.debugger.save_image(labeled_corners, "Ordered_Corners")

        # 4. Warp to a top-down chessboard
        warped, inverse_tf = self.transformer.warp_image(self.original_image, ordered)
        self.debugger.save_image(warped, "Warped_ChessBoard")

        # 5. Draw a precise grid on the warped image
        grid_warped = self.transformer.draw_precise_grid(warped, divisions=self.grid_divisions)
        self.debugger.save_image(grid_warped, "Warped_ChessBoard_Grid")

        # 6. Warp the grid back onto the original perspective
        grid_overlay = self.transformer.warp_grid_back(grid_warped, inverse_tf, self.original_image.shape)
        self.debugger.save_image(grid_overlay, "Grid_WarpedBack_Overlay")

        combined = self.transformer.overlay_grid(self.original_image, grid_overlay)
        self.debugger.save_image(combined, "Original_With_GridOverlay")

        # 7. Extract individual squares from the warped board
        squares = self.square_extractor.extract_squares(grid_warped)
        self.debugger.log(f"Extracted {len(squares)} squares.", level=0)

        return squares


# =============================================
#                EXAMPLE USAGE
# =============================================
if __name__ == "__main__":
    # Sample usage of the pipeline.
    # Adjust parameters as needed for your setup.
    detector = ChessboardDetector(
        image_path="Images/F.jpg",
        model_path="Models/ChessBoardCornersPredictor.pt",
        save_directory="Predictions",
        centroid_min_distance=50.0,
        grid_divisions=8,
        warp_chessboard_size=800,
        debug_level=2  # Increase to see more debug logs and detailed info
    )
    all_squares = detector.cut_squares()
    print("Squares extracted:", list(all_squares.keys()))