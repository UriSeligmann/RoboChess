import os
import cv2
import numpy as np

input_folder = "cropped_squares"
debug_folder = "Piece_Recognizer_Debug"
cache_file = "board_state_cache.npy"  # Add this line

# Create debug folder if it doesn't exist
os.makedirs(debug_folder, exist_ok=True)

MIN_PIECE_AREA = 25       # minimum area (in pixels) to consider as a "piece"
DISPLAY_HEIGHT = 200      # Reduced to fit more debug images

    # ───── CLAHE and Edge Detection parameters ─────
CLAHE_CLIP_LIMIT = 2.0      # Contrast limiting for CLAHE
CLAHE_TILE_SIZE = (8, 8)    # Grid size for CLAHE
CANNY_LOW = 30              # Lower threshold for Canny edge detection
CANNY_HIGH = 200            # Higher threshold for Canny edge detection
DILATION_KERNEL = (3, 3)    # Kernel size for dilation
DILATION_ITERATIONS = 4     # Number of dilation iterations

# ───── Color‐boost factors (optional) ─────
SATURATION_BOOST = 3
VALUE_BOOST = 3

# ───── Hue‐ranges for "abnormal" piece colors ─────
PINK_HUE_RANGE   = (80, 140)
YELLOW_HUE_RANGE = (15, 70)
MIN_SATURATION   = 70            # minimum S for a cluster to qualify
MIN_REGION_RATIO = 0.15  # minimum ratio of pixels for a significant region

def add_text_to_image(img, text, position=(10, 30), color=(255, 255, 255), thickness=1):
    """Add text with black outline for better visibility"""
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), thickness+1)
    cv2.putText(img, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, thickness)
    return img

def create_debug_grid(debug_images, filename):
    """Create a grid layout of debug images"""
    if not debug_images:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    
    # Resize all images to the same height for consistency
    target_height = DISPLAY_HEIGHT
    cols = 3
    rows = (len(debug_images) + cols - 1) // cols
    
    # First pass: resize to target height and find max width for entire grid
    resized_images = []
    max_width = 0
    
    for img in debug_images:
        if len(img.shape) == 2:  # Grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        h, w = img.shape[:2]
        new_width = int(w * target_height / h)
        max_width = max(max_width, new_width)
        resized = cv2.resize(img, (new_width, target_height))
        resized_images.append(resized)
    
    # Second pass: ensure all images have same width
    normalized_images = []
    for img in resized_images:
        if img.shape[1] != max_width:
            img = cv2.resize(img, (max_width, target_height))
        normalized_images.append(img)
    
    # Create grid layout
    grid_rows = []
    for r in range(rows):
        start_idx = r * cols
        end_idx = min((r + 1) * cols, len(normalized_images))
        row_images = normalized_images[start_idx:end_idx]
        
        # Pad last row if needed
        while len(row_images) < cols:
            padding = np.zeros((target_height, max_width, 3), dtype=np.uint8)
            row_images.append(padding)
        
        grid_rows.append(np.hstack(row_images))
    
    if grid_rows:
        final_grid = np.vstack(grid_rows)
        title_height = 40
        title_img = np.zeros((title_height, final_grid.shape[1], 3), dtype=np.uint8)
        cv2.putText(title_img, f"Debug Analysis: {filename}", (10, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return np.vstack([title_img, final_grid])
    
    return np.zeros((100, 100, 3), dtype=np.uint8)

def create_board_summary(size=400):
    """Create a chessboard summary visualization"""
    # Create empty board
    board_img = np.zeros((size, size, 3), dtype=np.uint8)
    square_size = size // 8
    
    # Draw chess pattern
    for row in range(8):
        for col in range(8):
            x1, y1 = col * square_size, row * square_size
            x2, y2 = x1 + square_size, y1 + square_size
            color = (200, 200, 200) if (row + col) % 2 == 0 else (100, 100, 100)
            cv2.rectangle(board_img, (x1, y1), (x2, y2), color, -1)
            
            # Draw square label
            file = chr(ord('A') + col)
            rank = str(8 - row)
            label = f"{file}{rank}"
            cv2.putText(board_img, label, (x1 + 5, y1 + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    return board_img

def analyze_region_color(hsv_roi, mask):
    """Analyze the dominant color in a masked region"""
    # Extract HSV values for the masked region
    masked_pixels = hsv_roi[mask > 0]
    if len(masked_pixels) == 0:
        return None, 0, 0, 0
    
    # Calculate mean HSV values
    mean_h = np.mean(masked_pixels[:, 0])
    mean_s = np.mean(masked_pixels[:, 1])
    mean_v = np.mean(masked_pixels[:, 2])
    
    # Determine color category
    color_name = "other"
    if mean_s >= MIN_SATURATION:
        if PINK_HUE_RANGE[0] <= mean_h <= PINK_HUE_RANGE[1]:
            color_name = "PINK"
        elif YELLOW_HUE_RANGE[0] <= mean_h <= YELLOW_HUE_RANGE[1]:
            color_name = "YELLOW"
    
    return color_name, mean_h, mean_s, mean_v

board_summary = create_board_summary()
piece_colors = {
    'PINK': (180, 105, 255),   # Pink in BGR
    'YELLOW': (0, 255, 255)    # Yellow in BGR
}

detected_pieces = {}

for filename in os.listdir(input_folder):
    if not filename.lower().endswith(".png"):
        continue

    img_path = os.path.join(input_folder, filename)
    image = cv2.imread(img_path)
    if image is None:
        print(f"Failed to load: {filename}")
        continue

    h, w = image.shape[:2]
    print(f"\n=== Processing {filename} ===")

    # ── DEBUG STEP 1: Original image ──
    debug_original = image.copy()
    add_text_to_image(debug_original, "1. Original", (10, 20))

    # ── DEBUG STEP 2: ROI cropping ──
    x0 = int(0.15 * w)
    x1 = int(0.85 * w)
    y0 = int(0.15 * h)
    y1 = int(0.85 * h)
    roi_bgr = image[y0:y1, x0:x1]
    
    debug_roi = image.copy()
    cv2.rectangle(debug_roi, (x0, y0), (x1, y1), (0, 255, 0), 2)
    add_text_to_image(debug_roi, "2. ROI Selection", (10, 20))

    # ── DEBUG STEP 3: HSV conversion & boosting ──
    hsv_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    debug_hsv_original = cv2.cvtColor(hsv_roi.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    hsv_roi[:, :, 1] = np.clip(hsv_roi[:, :, 1] * SATURATION_BOOST, 0, 255)
    hsv_roi[:, :, 2] = np.clip(hsv_roi[:, :, 2] * VALUE_BOOST, 0, 255)
    hsv_roi = hsv_roi.astype(np.uint8)
    
    debug_hsv_boosted = cv2.cvtColor(hsv_roi, cv2.COLOR_HSV2BGR)
    add_text_to_image(debug_hsv_original, "3a. HSV Original", (5, 15))
    add_text_to_image(debug_hsv_boosted, "3b. HSV Boosted", (5, 15))

    # Add blurring step
    kernel_size = (13, 13)  # adjust this for more/less blur
    hsv_roi = cv2.GaussianBlur(hsv_roi, kernel_size, 0)
    debug_blurred = cv2.cvtColor(hsv_roi, cv2.COLOR_HSV2BGR)
    add_text_to_image(debug_blurred, "3c. Blurred", (5, 15))

    # ── DEBUG STEP 4: CLAHE + Canny Edge Detection + Dilation ──
    # Convert to grayscale
    gray_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP_LIMIT, tileGridSize=CLAHE_TILE_SIZE)
    gray_clahe = clahe.apply(gray_roi)
    
    # Apply Canny edge detection
    edges = cv2.Canny(gray_clahe, CANNY_LOW, CANNY_HIGH)
    
    # Dilate edges to strengthen and close gaps
    kernel = np.ones(DILATION_KERNEL, np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=DILATION_ITERATIONS)
    
    # Debug visualizations
    debug_clahe = cv2.cvtColor(gray_clahe, cv2.COLOR_GRAY2BGR)
    debug_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    debug_dilated = cv2.cvtColor(edges_dilated, cv2.COLOR_GRAY2BGR)
    add_text_to_image(debug_clahe, "4a. CLAHE Enhanced", (5, 15))
    add_text_to_image(debug_edges, "4b. Canny Edges", (5, 15))
    add_text_to_image(debug_dilated, "4c. Dilated Edges", (5, 15))
    
    # Find contours in the dilated edges
    contours, _ = cv2.findContours(edges_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (most dominant shape)
    valid_regions = []
    total_area = hsv_roi.shape[0] * hsv_roi.shape[1]
    
    if contours:
        # Filter contours by area first
        large_contours = [c for c in contours if cv2.contourArea(c) > MIN_PIECE_AREA]
        
        if large_contours:
            # Get the largest contour
            largest_contour = max(large_contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            
            # Create mask for the largest contour
            mask = np.zeros(hsv_roi.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            
            # Create a slightly smaller ROI to ignore border regions
            border_margin = 2
            roi_height, roi_width = mask.shape[:2]
            roi_mask = np.zeros_like(mask)
            roi_mask[border_margin:roi_height-border_margin, 
                    border_margin:roi_width-border_margin] = 255
            
            # Apply ROI mask to ignore border regions
            mask = cv2.bitwise_and(mask, roi_mask)
            
            # Only proceed if the masked region is still significant
            if np.sum(mask > 0) > MIN_PIECE_AREA:
                # Analyze color in this region
                color_name, mean_h, mean_s, mean_v = analyze_region_color(hsv_roi, mask)
                ratio = area / total_area
                
                valid_regions.append({
                    'contour': largest_contour,
                    'area': area,
                    'ratio': ratio,
                    'color_name': color_name,
                    'mean_h': mean_h,
                    'mean_s': mean_s,
                    'mean_v': mean_v,
                    'mask': mask
                })
    
    # Debug visualization of contour detection
    debug_contours = cv2.cvtColor(edges_dilated, cv2.COLOR_GRAY2BGR)
    if valid_regions:
        region = valid_regions[0]  # Only one region (largest contour)
        color = (0, 255, 0) if region['color_name'] in ['PINK', 'YELLOW'] else (128, 128, 128)
        cv2.drawContours(debug_contours, [region['contour']], -1, color, 2)
        # Also show the filled mask
        debug_contours[region['mask'] > 0] = [color[0]//3, color[1]//3, color[2]//3]
    
    add_text_to_image(debug_contours, "4d. Largest Contour", (5, 15))

    # ── DEBUG STEP 5: Color analysis ──
    target_regions = []
    region_info = []
    
    for i, region in enumerate(valid_regions):
        color_name = region['color_name']
        mean_h, mean_s, mean_v = region['mean_h'], region['mean_s'], region['mean_v']
        ratio = region['ratio']
        
        is_target = False
        if color_name in ['PINK', 'YELLOW'] and ratio >= MIN_REGION_RATIO:
            target_regions.append(region)
            is_target = True
        
        region_info.append(f"R{i}: H={mean_h:3.0f} S={mean_s:3.0f} V={mean_v:3.0f} ({ratio:.1%}) -> {color_name}")
        print(f"  Region {i}: H={mean_h:3.0f}, S={mean_s:3.0f}, V={mean_v:3.0f} ({ratio:.1%}) -> {color_name} {'(TARGET)' if is_target else ''}")

    debug_color_analysis = debug_contours.copy()
    y_offset = 15
    for info in region_info:
        add_text_to_image(debug_color_analysis, info, (5, y_offset), (255, 255, 255), 1)
        y_offset += 15
    add_text_to_image(debug_color_analysis, "5. Color Analysis", (5, 5), (0, 255, 255), 1)

    # ── DEBUG STEP 6: Final result ──
    debug_final = debug_original.copy()
    piece_detected = False
    detected_color = None

    if target_regions:
        # Find the dominant target region (largest area)
        dominant_region = max(target_regions, key=lambda r: r['ratio'])
        detected_color = dominant_region['color_name']
        piece_detected = True

    if piece_detected:
        result_text = f"{detected_color} PIECE DETECTED"
        result_color = (0, 255, 0)
        print(f"  -> RESULT: {detected_color} piece detected!")
        cv2.rectangle(debug_final, (x0, y0), (x1, y1), (0, 255, 0), 2)
        detected_pieces[filename] = detected_color
    else:
        result_text = "NO PIECE DETECTED"
        print(f"  -> RESULT: No piece detected")
        result_color = (0, 0, 255)

    add_text_to_image(debug_final, result_text, (10, 30), result_color, 2)
    add_text_to_image(debug_final, "6. Final Result", (10, 10), (255, 255, 255), 1)

    # ── COMBINE ALL DEBUG IMAGES ──
    debug_images = [
        debug_original,      # 1. Original
        debug_roi,           # 2. ROI selection
        debug_hsv_original,  # 3a. HSV original
        debug_hsv_boosted,   # 3b. HSV boosted
        debug_blurred,       # 3c. Blurred
        debug_clahe,         # 4a. CLAHE enhanced
        debug_edges,         # 4b. Canny edges
        debug_dilated,       # 4c. Dilated edges
        debug_contours,      # 4d. Largest contour
        debug_color_analysis, # 5. Color analysis
        debug_final         # 6. Final result
    ]

    combined_debug = create_debug_grid(debug_images, filename)
    
    # Save debug image in the debug folder
    debug_path = os.path.join(debug_folder, f"Debug_{filename}")
    cv2.imwrite(debug_path, combined_debug)
               
# Update board summary with detected pieces
board_summary = create_board_summary()

# Initialize board state array (8x8) with None values
board_state = np.full((8, 8), None)

for filename, color in detected_pieces.items():
    # Extract board position from filename (assuming format like 'A1.png')
    if not filename.lower().endswith('.png'):
        continue
    
    file = ord(filename[0].upper()) - ord('A')  # 0-7 for A-H
    rank = 8 - int(filename[1])  # 0-7 for 8-1
    
    # Update board state array ('W' for pink, 'B' for yellow)
    piece_symbol = 'W' if color == 'PINK' else 'B' if color == 'YELLOW' else None
    board_state[rank, file] = piece_symbol
    
    # Calculate circle position
    square_size = board_summary.shape[0] // 8
    circle_x = file * square_size + square_size // 2
    circle_y = rank * square_size + square_size // 2
    radius = square_size // 3
    
    # Draw circle for detected piece
    piece_color = piece_colors.get(color, (128, 128, 128))
    cv2.circle(board_summary, (circle_x, circle_y), radius, piece_color, -1)
    cv2.circle(board_summary, (circle_x, circle_y), radius, (0, 0, 0), 2)

print("\nBoard State Array (W=Pink, B=Yellow, None=Empty):")
print(board_state)

# Save board state to cache file
np.save(cache_file, board_state)
print(f"\nBoard state saved to {cache_file}")

# Show the final board summary
cv2.imshow('Board Summary', board_summary)
cv2.waitKey(0)
cv2.destroyAllWindows()