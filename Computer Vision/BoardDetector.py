import cv2
import os
import numpy as np
from typing import Dict

from BoardDetector.wrapper_decorators import timeit_method, debug_entry_exit_method

from Debugger import Debugger
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

        # Shared debugger instance
        self.debugger = Debugger(debug_level, save_directory)

        # Sub-component initialization
        self.yolo_model = YOLOModel(model_path, self.debugger)
        self.centroid_processor = CentroidProcessor(centroid_min_distance, self.debugger)
        self.transformer = PerspectiveTransformer(warp_chessboard_size, self.debugger)
        self.grid_divisions = grid_divisions

    @timeit_method(level=1)
    @debug_entry_exit_method(level=2)
    def get_grid_overlay(self) -> np.ndarray:
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

        grid_overlay = self.transformer.warp_grid_back(
            grid_warped, 
            inverse_tf, 
            self.original_image.shape
        )
        self.debugger.save_image(grid_overlay, "Grid_WarpedBack_Overlay")

        mask = cv2.inRange(grid_overlay, (0, 255, 0), (0, 255, 0))
        return mask


# =============================================
#                MAIN EXECUTION
# =============================================
if __name__ == "__main__":

    import requests

    # URL of the ESP32-CAM snapshot endpoint
    url = "http://192.168.0.199/capture"

    """
    # Send GET request to capture a still image
    response = requests.get(url)

    # Check for success
    if response.status_code == 200:
        # Convert byte content to NumPy array
        img_array = np.frombuffer(response.content, dtype=np.uint8)
        # Decode image using OpenCV
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        cv2.imwrite("Capture.jpg", img)

    else:
        print(f"Failed to capture image. HTTP Status Code: {response.status_code}")
    """

    detector = ChessboardDetector(
        image_path=r"C:\Users\urise\OneDrive\Desktop\Robotics\Chess\Capture.jpg",
        model_path="Models/ChessBoardCornersPredictor.pt",
        save_directory="Board_Detector_Debug",
        centroid_min_distance=0.1,
        grid_divisions=8,
        warp_chessboard_size=800,
        debug_level=2
    )
    grid_overlay = detector.get_grid_overlay()
    cv2.imwrite("grid_overlay.png", grid_overlay)