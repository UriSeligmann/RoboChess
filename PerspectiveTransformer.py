import cv2
import numpy as np
from skimage.transform import ProjectiveTransform, warp

from typing import List, Tuple, Optional

from Wrapper_Decorators import debug_entry_exit_method, timeit_method
from Debugger import Debugger

# =============================================
#  PERSPECTIVE TRANSFORMATION & GRID OVERLAY
# =============================================
class PerspectiveTransformer:
    """
    Applies perspective transformations to images using scikit-image's ProjectiveTransform.
    Also provides methods to overlay and warp grids.I
    """
    def __init__(self, warp_size: int = 800, debugger: Optional[Debugger] = None) -> None:
        """
        Args:
            warp_size (int): Output size of the warped image (width=height).
            debugger (Debugger, optional): Debugger for logging.
        """
        self.warp_size = warp_size
        self.debugger = debugger

    def estimate_transform(self, ordered_corners: List[Tuple[float, float]]) -> ProjectiveTransform:
        """
        Estimates the projective transform given ordered corners.

        Args:
            ordered_corners (List[Tuple[float, float]]): The 4 corners in the source image.

        Raises:
            ValueError: If 4 corners are not provided.

        Returns:
            ProjectiveTransform: The estimated transform from source -> warped.
        """
        if len(ordered_corners) != 4:
            raise ValueError("Exactly 4 ordered corners are required for homography.")

        # Destination corners for an (warp_size x warp_size) output
        dst = np.array([
            [0, 0],
            [self.warp_size - 1, 0],
            [self.warp_size - 1, self.warp_size - 1],
            [0, self.warp_size - 1]
        ], dtype=np.float32)
        src = np.array(ordered_corners, dtype=np.float32)

        transform = ProjectiveTransform()
        success = transform.estimate(src, dst)
        if not success:
            raise RuntimeError("ProjectiveTransform estimation failed.")
        return transform

    @timeit_method(level=1)
    @debug_entry_exit_method(level=2)
    def warp_image(self, 
                image: np.ndarray,
                ordered_corners: List[Tuple[float, float]]
                ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Warps the given image to a top-down view based on the ordered corners using OpenCV.

        Args:
            image (np.ndarray): The original image (e.g., BGR or RGB).
            ordered_corners (List[Tuple[float, float]]): 4 corners in [TL, TR, BR, BL] order,
                specified as (x, y) coordinates within the original image.

        Returns:
            Tuple[np.ndarray, np.ndarray]:
                - warped_image (np.ndarray): The top-down (warped) image of size (warp_size, warp_size).
                - transform_matrix (np.ndarray): The 3x3 perspective transform matrix from the source
                image to the warped image.
        """
        src = np.array(ordered_corners, dtype=np.float32)
        warp_size = 800

        # destination corners for a warp_size x warp_size output
        dst = np.array([
            [0, 0],
            [warp_size - 1, 0],
            [warp_size - 1, warp_size - 1],
            [0, warp_size - 1]
        ], dtype=np.float32)

        # Compute the perspective transform matrix
        M = cv2.getPerspectiveTransform(src, dst)

        # Warp the image to a top-down view
        warped = cv2.warpPerspective(image, M, (warp_size, warp_size))

        return warped, M


    def draw_precise_grid(self, image: np.ndarray, divisions: int = 8) -> np.ndarray:
        """
        Draws a grid on the image with the specified number of divisions.

        Args:
            image (np.ndarray): The image on which to draw the grid.
            divisions (int): Number of grid divisions (both horizontal and vertical).

        Returns:
            np.ndarray: A copy of the image with green grid lines drawn.
        """
        grid_img = image.copy()
        h, w = grid_img.shape[:2]
        thickness = 3
        for i in range(1, divisions):
            x = int(w * i / divisions)
            cv2.line(grid_img, (x, 0), (x, h), (0, 255, 0), thickness)
        for i in range(1, divisions):
            y = int(h * i / divisions)
            cv2.line(grid_img, (0, y), (w, y), (0, 255, 0), thickness)
        return grid_img

    def warp_grid_back(self, 
                       grid_image: np.ndarray, 
                       inverse_transform: ProjectiveTransform, 
                       original_shape: Tuple[int, int, int]
                      ) -> np.ndarray:
        """
        Warps a grid image back to the original perspective using the inverse transform.

        Args:
            grid_image (np.ndarray): The warped grid image.
            inverse_transform (ProjectiveTransform): The transform from warped -> source.
            original_shape (Tuple[int,int,int]): The original image shape (H, W, C).

        Returns:
            np.ndarray: Grid image in the original perspective.
        """
        grid_float = grid_image.astype(np.float32) / 255.0
        h, w = original_shape[:2]
        # Warp back
        unwarped_float = warp(grid_float, inverse_transform, output_shape=(h, w))
        unwarped = (unwarped_float * 255.0).astype(np.uint8)
        return unwarped

    def overlay_grid(self, 
                     original_image: np.ndarray, 
                     grid_overlay: np.ndarray
                    ) -> np.ndarray:
        """
        Overlays a grid (e.g., red lines) onto the original image.

        Args:
            original_image (np.ndarray): The source image.
            grid_overlay (np.ndarray): The grid image aligned to the original perspective.

        Returns:
            np.ndarray: The combined overlay.
        """
        grid_gray = cv2.cvtColor(grid_overlay, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(grid_gray, 10, 255, cv2.THRESH_BINARY)
        grid_color = np.zeros_like(original_image, dtype=np.uint8)
        grid_color[:] = (0, 0, 255)  # Red overlay for the grid
        # Merge with mask
        grid_masked = cv2.bitwise_and(grid_color, grid_color, mask=mask)
        combined = cv2.addWeighted(original_image, 1.0, grid_masked, 1.0, 0)
        return combined