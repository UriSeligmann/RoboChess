import cv2
import numpy as np
from skimage.transform import ProjectiveTransform
from typing import List, Tuple, Optional
import os

# =============================================
#  PERSPECTIVE TRANSFORMATION & GRID OVERLAY
# =============================================
class PerspectiveTransformer:
    """
    Applies perspective transformations to images using scikit-image's ProjectiveTransform.
    Also provides methods to overlay and warp grids.I
    """
    def __init__(self, warp_size: int = 800) -> None:
        """
        Args:
            warp_size (int): Output size of the warped image (width=height).
        """
        self.warp_size = warp_size

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
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - warped_image (np.ndarray): The top-down (warped) image of size (warp_size, warp_size).
                - transform_matrix (np.ndarray): The 3x3 perspective transform matrix from the source
                image to the warped image.
                - src_corners (np.ndarray): Source corner points used for transformation.
                - dst_corners (np.ndarray): Destination corner points used for transformation.
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

        return warped, M, src, dst

    def save_transform_matrix(self, 
                             transform_matrix: np.ndarray, 
                             filepath: str,
                             create_dirs: bool = True,
                             src_corners: Optional[np.ndarray] = None,
                             dst_corners: Optional[np.ndarray] = None) -> None:
        """
        Saves the perspective transform matrix as a .npy file for later use.

        Args:
            transform_matrix (np.ndarray): The 3x3 perspective transform matrix to save.
            filepath (str): Path where to save the matrix. Should end with .npy extension.
            create_dirs (bool): Whether to create parent directories if they don't exist.
            src_corners (np.ndarray, optional): Source corner points used to compute the transform.
            dst_corners (np.ndarray, optional): Destination corner points used to compute the transform.

        Raises:
            ValueError: If the transform matrix is not 3x3.
            OSError: If there are issues with file writing or directory creation.
        """
        if transform_matrix.shape != (3, 3):
            raise ValueError(f"Transform matrix must be 3x3, got shape {transform_matrix.shape}")
        
        # Ensure filepath has .npy extension
        if not filepath.endswith('.npy'):
            filepath += '.npy'
        
        # Create parent directories if requested
        if create_dirs:
            parent_dir = os.path.dirname(filepath)
            if parent_dir and not os.path.exists(parent_dir):
                os.makedirs(parent_dir, exist_ok=True)
        
        try:
            # Create data dictionary with matrix and optional corner points
            data_to_save = {'transform_matrix': transform_matrix}
            if src_corners is not None:
                data_to_save['src_corners'] = src_corners
            if dst_corners is not None:
                data_to_save['dst_corners'] = dst_corners
            
            np.savez(filepath.replace('.npy', '.npz'), **data_to_save)
        except Exception as e:
            error_msg = f"Failed to save transform matrix to {filepath}: {str(e)}"
            raise OSError(error_msg)

    def load_transform_matrix(self, filepath: str) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Loads a perspective transform matrix from a .npz file.

        Args:
            filepath (str): Path to the .npz file containing the transform data.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]: 
                - transform_matrix: The 3x3 perspective transform matrix
                - src_corners: Source corner points (if saved)
                - dst_corners: Destination corner points (if saved)

        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the loaded matrix is not 3x3.
            OSError: If there are issues with file reading.
        """
        # Try .npz first, fallback to .npy for backward compatibility
        npz_path = filepath.replace('.npy', '.npz')
        if not os.path.exists(npz_path) and not os.path.exists(filepath):
            raise FileNotFoundError(f"Transform file not found: {npz_path} or {filepath}")
        
        try:
            if os.path.exists(npz_path):
                data = np.load(npz_path)
                transform_matrix = data['transform_matrix']
                src_corners = data.get('src_corners', None)
                dst_corners = data.get('dst_corners', None)
            else:
                # Fallback to old .npy format
                transform_matrix = np.load(filepath)
                src_corners = dst_corners = None
            
            if transform_matrix.shape != (3, 3):
                raise ValueError(f"Loaded matrix must be 3x3, got shape {transform_matrix.shape}")
            
            return transform_matrix, src_corners, dst_corners
            
        except Exception as e:
            error_msg = f"Failed to load transform matrix from {filepath}: {str(e)}"
            raise OSError(error_msg)

    def warp_image_with_saved_matrix(self, 
                                   image: np.ndarray, 
                                   matrix_filepath: str) -> np.ndarray:
        """
        Warps an image using a previously saved transform matrix.

        Args:
            image (np.ndarray): The image to warp.
            matrix_filepath (str): Path to the .npy file containing the transform matrix.

        Returns:
            np.ndarray: The warped image.
        """
        transform_matrix, _, _ = self.load_transform_matrix(matrix_filepath)
        warped = cv2.warpPerspective(image, transform_matrix, (self.warp_size, self.warp_size))
        return warped

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
        thickness = 10
        for i in range(0, divisions+1):
            x = int(w * i / divisions)
            cv2.line(grid_img, (x, 0), (x, h), (0, 255, 0), thickness)
        for i in range(0, divisions+1):
            y = int(h * i / divisions)
            cv2.line(grid_img, (0, y), (w, y), (0, 255, 0), thickness)
        return grid_img