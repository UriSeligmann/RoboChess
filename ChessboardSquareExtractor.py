import os
import cv2
import numpy as np
from typing import Dict, Tuple

from Wrapper_Decorators import debug_entry_exit_method


# =============================================
#     CHESSBOARD (GRID) SQUARE EXTRACTION
# =============================================
class ChessboardSquareExtractor:
    """
    Splits a chessboard (or any grid) image into individual cells by detecting
    the green grid lines. This assumes:
      1) The chessboard is on a black background (pure zeros).
      2) Bright green lines are drawn on the board.
    """
    def __init__(self, 
                 save_directory: str = "Predictions",
                 columns: int = 8,
                 rows: int = 8) -> None:
        """
        Args:
            save_directory (str): Directory to save extracted squares.
            columns (int): Number of columns on the board (e.g., 8 for standard chess).
            rows (int): Number of rows on the board (e.g., 8 for standard chess).
        """
        self.save_directory = save_directory
        os.makedirs(self.save_directory, exist_ok=True)

        # If using standard chess, we'll label columns as A..H, rows as 1..8, etc.
        # Adjust if you prefer different labeling or a bigger board.
        self.column_labels = [chr(ord('A') + i) for i in range(columns)]
        self.row_labels = [str(i + 1) for i in range(rows)]
        # We expect (columns + 1) vertical lines, (rows + 1) horizontal lines.
        self.expected_vertical_lines = columns + 1
        self.expected_horizontal_lines = rows + 1

    @debug_entry_exit_method(level=2)
    def extract_squares(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extracts each square from the given image, which must have:
         - black background (pixels = 0)
         - green grid lines drawn on the board.

        Steps:
          1) Crop out black padding.
          2) Create a mask of the green lines.
          3) Morphologically separate horizontal and vertical lines.
          4) Find coordinates of these lines.
          5) Slice out each cell between consecutive lines.
          6) Label each cell ("A1", "B1", etc.) and save to disk.

        Args:
            image (np.ndarray): BGR image with black background and green lines.
        
        Returns:
            Dict[str, np.ndarray]: A dictionary mapping square-label (e.g. 'A1')
                                   to the sub-image (square).
        """
        # 1) Crop black border
        board_img = self._crop_black_border(image)

        # 2) Create mask for green lines
        green_mask = self._get_green_mask(board_img)

        cv2.imshow("Green Mask", green_mask)
        cv2.waitKey(0)


        # 3) Separate into horizontal and vertical line masks
        horizontal_lines, vertical_lines = self._split_lines(green_mask)

        # 4) Get y-coordinates for horizontal lines, x-coordinates for vertical lines
        y_coords = self._find_line_coordinates(horizontal_lines, axis='horizontal')
        x_coords = self._find_line_coordinates(vertical_lines, axis='vertical')

        # 5) Slice out each square
        squares = self._slice_squares(board_img, x_coords, y_coords)

        # 6) Save each square and return dictionary
        for label, sq in squares.items():
            out_path = os.path.join(self.save_directory, f"Square_{label}.jpg")
            cv2.imwrite(out_path, sq)

        return squares

    # -------------------------------------------------------------------------
    #                         Helper methods
    # -------------------------------------------------------------------------
    def _crop_black_border(self, image: np.ndarray) -> np.ndarray:
        """
        Finds the bounding box of non-zero pixels in `image` and returns a cropped sub-image.
        Assumes pure-black background is truly 0.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Locate all non-black (non-zero) points
        coords = cv2.findNonZero(gray)
        if coords is None:
            # If nothing but black, return as-is
            return image

        x, y, w, h = cv2.boundingRect(coords)
        return image[y:y+h, x:x+w]

    def _get_green_mask(self, board_img: np.ndarray) -> np.ndarray:
        """
        Threshold the board image in RGB to isolate the bright green lines.
        """

        # Adjust these bounds based on your actual green color
        lower_green = np.array([0, 255, 0], dtype=np.uint8)
        upper_green = np.array([0, 255, 0], dtype=np.uint8)

        mask = cv2.inRange(board_img, lower_green, upper_green)
        return mask

    def _split_lines(self, green_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Use morphology to separate horizontal and vertical lines.
        """
        # For horizontal lines (wide kernel, 1-pixel tall)
        horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 1))
        horizontal_lines = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, horiz_kernel)

        # For vertical lines (tall kernel, 1-pixel wide)
        vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 30))
        vertical_lines = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, vert_kernel)

        return horizontal_lines, vertical_lines

    def _find_line_coordinates(self, bin_image: np.ndarray, axis: str) -> np.ndarray:
        """
        Given a binary mask of lines, return a sorted list of unique line coordinates.
        For axis='horizontal', return y-coordinates.
        For axis='vertical', return x-coordinates.
        """
        coords = []
        contours, _ = cv2.findContours(bin_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if axis == 'horizontal':
                # The center row of the line
                line_coord = y + h // 2
            else:  # vertical
                # The center column of the line
                line_coord = x + w // 2
            coords.append(line_coord)

        coords.sort()

        # Merge near duplicates in case lines are thick or overlap
        merged = []
        min_gap = 5
        for c in coords:
            if not merged or abs(c - merged[-1]) > min_gap:
                merged.append(c)

        return np.array(merged, dtype=int)

    def _slice_squares(self,
                       board_img: np.ndarray,
                       x_coords: np.ndarray,
                       y_coords: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Slices out each square using the sorted sets of x- and y-coordinates.
        Returns a dict { 'A1': cell_img, 'A2': cell_img, ... }.
        Assumes x_coords and y_coords each have one more line than there are squares.
        """
        squares = {}
        # For standard chess labeling, top row is '8' and bottom is '1'.
        # That means we invert row indexing.

        expected_h = self.expected_horizontal_lines  # e.g., 9 if 8 squares
        expected_v = self.expected_vertical_lines    # e.g., 9 if 8 squares

        # Basic check
        if len(x_coords) < expected_v or len(y_coords) < expected_h:
            print("[WARNING] The detected number of lines doesn't match the expected count. "
                  "Check your green mask or morphological settings.")
        
        # Number of squares in each dimension:
        # (e.g., if we have 9 lines horizontally, we have 8 squares tall)
        n_h_squares = len(y_coords) - 1
        n_v_squares = len(x_coords) - 1

        # We'll clamp to the lesser of what we found vs. what's expected
        n_h_squares = min(n_h_squares, len(self.row_labels))
        n_v_squares = min(n_v_squares, len(self.column_labels))

        for row_idx in range(n_h_squares):
            for col_idx in range(n_v_squares):
                y1 = y_coords[row_idx]
                y2 = y_coords[row_idx + 1]
                x1 = x_coords[col_idx]
                x2 = x_coords[col_idx + 1]

                # Crop the cell
                cell_img = board_img[y1:y2, x1:x2]

                # Build a label => columns left-to-right = A,B,C..., rows top-to-bottom = 8..1
                label_col = self.column_labels[col_idx]
                label_row = self.row_labels[n_h_squares - row_idx - 1]  # invert row indexing
                label = f"{label_col}{label_row}"

                squares[label] = cell_img

        return squares
