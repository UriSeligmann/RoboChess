# chess_pipeline.py
"""Complete end‑to‑end pipeline for detecting a physical chess position and
returning an updated FEN string.

Key components
--------------
* **ESP32‑CAM capture**  – optional; otherwise loads a local JPEG/PNG.
* **ChessboardDetector** – finds the board, rectifies it to 800×800, and
  stores all intermediate debug images in *Board_Detector_Debug*.
* **K-means colour segmentation** – automatically detects the two brightest colors
  for piece identification. Saves comprehensive debug images to *ColourSegmentation_Debug*.
* **MoveCalculator.difference_based_infer_fen** – uses the colour matrix plus
  a current FEN to infer the next position.

The public helper ``getFen()`` is what the rest of your application should
call.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import requests

from boardDetector import ChessboardDetector
from MoveCalculator import difference_based_infer_fen
from debug_utils.DebugManager import DebugManager

# ────────────────────────────────────────────────────────────────
#  Global debug directories – created at import time so they exist
# ────────────────────────────────────────────────────────────────
BOARD_DETECTOR_DIR = Path("Board_Detector_Debug")
COLOUR_DEBUG_DIR   = Path("ColourSegmentation_Debug")
BOARD_DETECTOR_DIR.mkdir(exist_ok=True)
COLOUR_DEBUG_DIR.mkdir(exist_ok=True)


# ===========================================================================
#  1️⃣  Utility: grab a frame directly from an ESP32‑CAM
# ===========================================================================

def capture_from_esp32(url: str = "http://192.168.1.50/capture") -> np.ndarray:
    """Snapshot the camera and return an OpenCV BGR image."""
    resp = requests.get(url, timeout=5)
    if resp.status_code != 200:
        raise ConnectionError(f"ESP32‑CAM request failed ({resp.status_code})")
    return cv2.imdecode(np.frombuffer(resp.content, np.uint8), cv2.IMREAD_COLOR)


def color_cluster(img: np.ndarray, k: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply K-means clustering to group similar colors together.
    
    Args:
        img: Input BGR image
        k: Number of clusters
        
    Returns:
        Tuple of (clustered_image, labels, centers)
    """
    Z = img.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(Z, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    clustered = centers[labels.flatten()].reshape(img.shape)
    return clustered, labels, centers


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Apply lighting correction and saturation enhancement to improve color detection.
    
    Args:
        img: Input BGR image
        
    Returns:
        Preprocessed image
    """
    # Step 1: Lighting correction with CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    lighting_corrected = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    
    # Step 2: Maximize saturation initially
    hsv = cv2.cvtColor(lighting_corrected, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = 255
    intensified = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Step 3: Gaussian blur to smooth colors
    blurred = cv2.GaussianBlur(intensified, (55, 55), 0)
    
    return blurred


def apply_morphological_operations(clustered: np.ndarray) -> np.ndarray:
    """
    Apply morphological operations to clean up the clustered image.
    
    Args:
        clustered: K-means clustered image
        
    Returns:
        Morphologically processed image
    """
    gray = cv2.cvtColor(clustered, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    morphed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    morphed = cv2.morphologyEx(morphed, cv2.MORPH_OPEN, kernel, iterations=1)
    morphed = cv2.dilate(morphed, kernel, iterations=1)
    
    clustered_morphed = cv2.bitwise_and(clustered, clustered, mask=morphed)
    
    # Boost saturation again for better visibility
    hsv_morphed = cv2.cvtColor(clustered_morphed, cv2.COLOR_BGR2HSV)
    hsv_morphed[..., 1] = 255
    clustered_morphed = cv2.cvtColor(hsv_morphed, cv2.COLOR_HSV2BGR)
    
    return clustered_morphed


def detect_brightest_colors(clustered_morphed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Automatically detect the two brightest colors in the processed image.
    
    Args:
        clustered_morphed: Morphologically processed clustered image
        
    Returns:
        Tuple of (mask1, mask2, combined_mask) for the two brightest colors
    """
    Z = clustered_morphed.reshape((-1, 3))
    unique_colors, counts = np.unique(Z, axis=0, return_counts=True)
    
    # Compute intensity of each unique color (sum of B+G+R)
    intensity = unique_colors.sum(axis=1)
    
    # Find indices of the two brightest colors
    brightest_idx = intensity.argsort()[-2:]  # two brightest
    brightest_colors = unique_colors[brightest_idx]
    
    # Create masks for the two brightest colors
    mask1 = cv2.inRange(clustered_morphed, brightest_colors[0], brightest_colors[0])
    mask2 = cv2.inRange(clustered_morphed, brightest_colors[1], brightest_colors[1])
    
    # Clean small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel, iterations=1)
    mask2 = cv2.morphologyEx(mask2, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Create color-coded mask for visualization
    combined_mask = np.zeros_like(clustered_morphed)
    combined_mask[mask1 > 0] = (180, 105, 255)  # pink for first brightest
    combined_mask[mask2 > 0] = (0, 255, 255)    # yellow for second brightest
    
    return mask1, mask2, combined_mask


def average_blob_area(*masks: np.ndarray) -> float:
    """Calculate average area of blobs across multiple masks."""
    areas = []
    for mask in masks:
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        for c in cnts:
            a = cv2.contourArea(c)
            if a > 0:
                areas.append(a)
    return np.mean(areas) if areas else 0


# ===========================================================================
#  2️⃣  K-means based colour segmentation on the warped 800×800 board image
# ===========================================================================

def extract_board_state_from_image(board_img: np.ndarray, debug: bool = True) -> np.ndarray:
    debugger = DebugManager(debug_level=1 if debug else 0, save_directory=str(COLOUR_DEBUG_DIR))
    
    # Step 1: Preprocess the image
    preprocessed = preprocess_image(board_img)
    debugger.save_image(preprocessed, "01_preprocessed")
    
    # Step 2: Apply K-means clustering
    clustered, labels, centers = color_cluster(preprocessed, k=8)
    debugger.save_image(clustered, "02_clustered")
    
    # Step 3: Apply morphological operations
    clustered_morphed = apply_morphological_operations(clustered)
    debugger.save_image(clustered_morphed, "03_clustered_morphed")
    
    # Step 4: Detect the two brightest colors
    mask1, mask2, combined_mask = detect_brightest_colors(clustered_morphed)
    debugger.save_image(mask1, "04_mask1_brightest")
    debugger.save_image(mask2, "05_mask2_second_brightest")
    debugger.save_image(combined_mask, "06_combined_mask")
    
    # Step 5: Create comparison visualization
    stacked_display = np.hstack((board_img, clustered_morphed, combined_mask))
    debugger.save_image(stacked_display, "07_comparison_original_clustered_masks")
    
    # Step 6: Analyze each square of the chessboard
    h, w = board_img.shape[:2]
    cell_h, cell_w = h // 8, w // 8
    avg_area = average_blob_area(mask1, mask2)
    threshold = max(avg_area * 0.3, 100)  # Minimum threshold to avoid noise
    
    board_state: np.ndarray = np.full((8, 8), None, dtype=object)
    overlay = board_img.copy()
    h, w = board_img.shape[:2]
    cell_h, cell_w = h // 8, w // 8
    cell_size = (cell_h, cell_w)
    
    # Draw the grid
    overlay = debugger.draw_grid(overlay, cell_size)
    
    for r in range(8):
        for c in range(8):
            y1, y2 = r * cell_h, (r + 1) * cell_h
            x1, x2 = c * cell_w, (c + 1) * cell_w
            
            # Crop into the central 85% to avoid edge artifacts
            inset_h = int(cell_h * 0.075)  # 7.5% from top & bottom
            inset_w = int(cell_w * 0.075)  # 7.5% from left & right
            cy1, cy2 = y1 + inset_h, y2 - inset_h
            cx1, cx2 = x1 + inset_w, x2 - inset_w
            
            # Count pixels in each mask within the cell
            mask1_cnt = cv2.countNonZero(mask1[cy1:cy2, cx1:cx2])
            mask2_cnt = cv2.countNonZero(mask2[cy1:cy2, cx1:cx2])
            
            centre = (x1 + cell_w // 2, y1 + cell_h // 2)
            radius = cell_w // 4
            
            # Determine piece type based on absolute pixel count threshold
            if mask1_cnt > threshold:
                board_state[r, c] = 'W'
                overlay = debugger.draw_piece_annotations(overlay, (r, c), 'W', cell_size)
            elif mask2_cnt > threshold:
                board_state[r, c] = 'B'
                overlay = debugger.draw_piece_annotations(overlay, (r, c), 'B', cell_size)
            
            overlay = debugger.add_coordinate_label(overlay, (r, c), cell_size)
    
    debugger.save_image(overlay, "08_annotated_board")
    
    if debug:
        debugger.log("K-means clustering completed")
        debugger.log(f"Average blob area: {avg_area:.1f}, threshold: {threshold:.1f}")
        debugger.log(f"Board matrix (top→bottom):\n{board_state}")
    
    return board_state


# ===========================================================================
#  3️⃣  Main pipeline function
# ===========================================================================

def process_chessboard(image_path: str = "capture.jpg", *, use_camera: bool = False,
                       debug: bool = True) -> np.ndarray:
    """Run the complete detection→segmentation pipeline and return the matrix."""

    # 0. Acquire or load image ------------------------------------------------
    if use_camera:
        cv2.imwrite(image_path, capture_from_esp32())
        if debug:
            print("Image captured and saved →", image_path)

    # 1. Board detection / warping -------------------------------------------
    if debug:
        print("\n=== Stage 1: Board Detection ===")
    detector = ChessboardDetector(
        image_path=image_path,
        model_path="Models/ChessBoardCornersPredictor.pt",
        save_directory=str(BOARD_DETECTOR_DIR),
        centroid_min_distance=0.1,
        grid_divisions=8,
        warp_chessboard_size=800,
        debug_level=1 if debug else 0,
    )
    
    board_img = detector.get_grid_overlay()  # triggers processing & saves debug images
    if board_img is None:
        raise FileNotFoundError("Warped board image not found – check ChessboardDetector output.")

    if debug:
        print("Board detected. Proceeding to K-means color analysis…")
        print("\n=== Stage 2: K-means Color Segmentation ===")

    # 2. K-means colour segmentation ------------------------------------------
    return extract_board_state_from_image(board_img, debug=debug)


# ===========================================================================
#  4️⃣  Public helper: current FEN → updated FEN
# ===========================================================================

def getFen(image_path: str, current_fen: str, *, debug: bool = True) -> str:
    """Convenience wrapper – returns the new FEN after analysing *image_path*."""
    board_state = process_chessboard(image_path, use_camera=False, debug=debug)

    color_board: List[List[Optional[str]]] = [
        ['B' if x == 'B' else 'W' if x == 'W' else None for x in row]
        for row in board_state
    ]

    if debug:
        print("\n=== Stage 3: FEN Generation ===")
        print("Colour board passed to MoveCalculator:")
        for i, row in enumerate(color_board):
            print(f"Rank {8-i}: {row}")

    return difference_based_infer_fen(current_fen, color_board)