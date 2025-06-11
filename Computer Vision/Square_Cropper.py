import cv2
import numpy as np
import os

class ChessboardCropper:
    def __init__(self, image_path, mask_path, output_folder="cropped_squares"):
        self.output_folder = output_folder
        self.files = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        self.labeled_polys = {}
        
        # Load images
        self.original_image = cv2.imread(image_path)
        self.mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        if self.mask is None:
            raise FileNotFoundError("Mask file not found.")
        if self.original_image is None:
            raise FileNotFoundError("Original file not found.")
            
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)

    def _polygon_center(self, poly):
        """Returns (cx, cy), the mean of polygon corner coords"""
        pts_2d = poly.reshape(-1, 2)
        return (np.mean(pts_2d[:, 0]), np.mean(pts_2d[:, 1]))

    def _group_polys_into_grid(self, polys, expected_rows=8, expected_cols=8):
        """Group polygons into a proper 8x8 grid with better tolerance and validation"""
        if len(polys) == 0:
            return []
        
        # Sort by Y coordinate first (top to bottom)
        polys.sort(key=lambda p: self._polygon_center(p)[1])
        
        # Calculate average polygon dimensions for better tolerance
        centers = [self._polygon_center(p) for p in polys]
        y_coords = [c[1] for c in centers]
        
        if len(y_coords) > expected_rows:
            y_range = max(y_coords) - min(y_coords)
            row_height = y_range / (expected_rows - 1) if expected_rows > 1 else y_range
            tolerance = max(row_height * 0.3, 15)
        else:
            tolerance = 20
        
        rows = []
        current_row = []
        last_y = None
        
        for p in polys:
            _, y = self._polygon_center(p)
            if last_y is None or abs(y - last_y) < tolerance:
                current_row.append(p)
                last_y = y if last_y is None else (last_y + y) / 2
            else:
                if current_row:
                    current_row.sort(key=lambda p: self._polygon_center(p)[0])
                    rows.append(current_row)
                current_row = [p]
                last_y = y
        
        if current_row:
            current_row.sort(key=lambda p: self._polygon_center(p)[0])
            rows.append(current_row)
        
        if expected_cols > 0:
            valid_rows = [row for row in rows if abs(len(row) - expected_cols) <= 1]
            if len(valid_rows) >= expected_rows - 2:
                rows = valid_rows[:expected_rows]
        
        return rows

    def _order_points(self, pts):
        """Reorders corners into [top-left, top-right, bottom-right, bottom-left]"""
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        diff = np.diff(pts, axis=1)
        
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect

    def process(self, debug=False):
        """Main processing method"""
        # Binary threshold
        _, binary_mask = cv2.threshold(self.mask, 127, 255, cv2.THRESH_BINARY)
        if debug:
            cv2.imshow("Binary Mask", binary_mask)
            cv2.waitKey(0)

        # Find and filter contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        approx_polys = []
        
        for cnt in contours:
            if cv2.contourArea(cnt) < 10:
                continue
            
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.1 * peri, True)
            
            if len(approx) == 4:
                x, y, w, h = cv2.boundingRect(approx)
                if w >= 5 and h >= 5:
                    approx_polys.append(approx)

        # Filter outliers
        areas = [cv2.contourArea(poly) for poly in approx_polys]
        if areas:
            mean_area = np.mean(areas)
            std_area = np.std(areas)
            bounds = (mean_area - 2 * std_area, mean_area + 2 * std_area)
            filtered_polys = [p for p in approx_polys if bounds[0] <= cv2.contourArea(p) <= bounds[1]]
        else:
            filtered_polys = []

        # Create debug image
        debug_image = self.original_image.copy()
        overlay = debug_image.copy()
        
        for poly in filtered_polys:
            cv2.fillPoly(overlay, [poly], (0, 255, 255))
        
        cv2.addWeighted(overlay, 0.4, debug_image, 0.6, 0, debug_image)

        # Organize into grid
        grid = self._group_polys_into_grid(filtered_polys)
        
        # Label squares
        for row_idx, row in enumerate(grid[:8]):
            for col_idx, poly in enumerate(row[:8]):
                if poly is None:
                    continue
                    
                rank = row_idx + 1
                file = self.files[7 - col_idx]
                label = f"{file}{rank}"

                cx, cy = self._polygon_center(poly)
                if debug:
                    cv2.polylines(debug_image, [poly], True, (0, 255, 0), 2)
                    cv2.putText(debug_image, label, (int(cx)-10, int(cy)+10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                
                self.labeled_polys[label] = poly

        if debug:
            cv2.imshow("Filtered Polygons on Original", debug_image)
            cv2.waitKey(0)

        # Crop and save squares
        for label, poly in self.labeled_polys.items():
            poly_mask = np.zeros(self.original_image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(poly_mask, [poly], 255)
            masked = cv2.bitwise_and(self.original_image, self.original_image, mask=poly_mask)
            x, y, w, h = cv2.boundingRect(poly)
            roi = masked[y:y+h, x:x+w]

            if w >= 5 and h >= 5:
                out_path = os.path.join(self.output_folder, f"{label}.png")
                cv2.imwrite(out_path, roi)
                if debug:
                    cv2.imshow(f"Square {label}", roi)
                    cv2.waitKey(0)
                    cv2.destroyWindow(f"Square {label}")

        if debug:
            cv2.destroyAllWindows()
            
        return self.labeled_polys

# Example usage:
if __name__ == "__main__":
    cropper = ChessboardCropper("Capture.jpg", "grid_overlay.png")
    cropper.process(debug=False)