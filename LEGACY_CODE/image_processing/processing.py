
import numpy as np
import cv2
import copy
import rasterio
import matplotlib.pyplot as plt
import json
import os
from collections import defaultdict
import glob
import exifread
from PIL import Image
from rasterio.plot import reshape_as_image
from reportlab.graphics.shapes import Drawing, Rect, Polygon, String
from reportlab.lib import colors
from reportlab.graphics import renderPDF
from reportlab.graphics.shapes import Rect





def annotate_and_downscale_orthophoto(
    ortho_path, 
    json_path, 
    output_path="ortho.png", 
    scale_factor=0.5
):
    """
    Annotates the orthophoto with defect locations (hotspots, faulty diodes, offline panels), 
    then downsizes the image.

    Args:
        ortho_path (str): Path to the orthophoto TIFF file.
        json_path (str): Path to the JSON file containing bounding boxes.
        output_path (str): Path to save the annotated and downscaled image.
        scale_factor (float): Factor to downscale the image (default: 0.5).
    """

    # Load JSON bounding boxes
    with open(json_path, "r") as f:
        data = json.load(f)
    
    bounding_boxes = data[0]["boundingBox"]["boundingBoxes"]

    # Load the TIFF image
    with rasterio.open(ortho_path) as src:
        ortho_img = np.moveaxis(src.read(), 0, -1).astype(np.uint8)  # Convert channels
        ortho_img = cv2.cvtColor(ortho_img, cv2.COLOR_RGB2BGR)  # Convert to OpenCV format

    # Define colors for annotation
    color_map = {
        "hotspots": (0, 0, 255),      # Red
        "faultydiodes": (255, 0, 0),  # Blue
        "offlinepanels": (0, 255, 0)  # Green
    }

    # Annotate bounding boxes (ignore solar panels)
    for bbox in bounding_boxes:
        left, top, width, height = bbox["left"], bbox["top"], bbox["width"], bbox["height"]
        label = bbox["label"]

        if label == "solarpanels":
            continue  # Skip solar panels

        if label in color_map:
            color = color_map[label]

            # Draw bounding box
            cv2.rectangle(ortho_img, (left, top), (left + width, top + height), color, thickness=3)

            # Add label text
            text_pos = (left, top - 5)
            cv2.putText(
                ortho_img, label, text_pos,
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, lineType=cv2.LINE_AA
            )

    # Downscale the image
    downscaled_img = cv2.resize(ortho_img, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)

    # Save the annotated orthophoto
    cv2.imwrite(output_path, downscaled_img)
    print(f"Annotated and downscaled orthophoto saved at {output_path}")

def generate_defect_map(
    tif_path,
    annotation_json_path,
    alignment='vertical',  # alignment parameter is no longer used for numbering
    output_image="annotated_defects_map.pdf"
):
    """
    Reads bounding boxes from a JSON annotation file, draws panels and defects 
    onto a vector PDF canvas using ReportLab, and organizes them into a panel-level dictionary.
    
    In this version:
      - The final output is a PDF file (vectorized) that can be included in LaTeX.
      - Panel areas with any issues are filled with red at 50% opacity.
      - Panels are numbered so that the TOP-LEFT panel is 1-1. (Note that the JSON and OpenCV
        use a top-left origin, but ReportLab uses a bottom-left origin, so the y values are transformed.)
    
    Returns:
        panel_defects_dict: Nested dictionary with structure:
          {
              (col, row): {
                  "bbox": (x, y, w, h),  # drawn coordinates (with y transformed)
                  "hotspots": [ {...}, ...],
                  "faultydiodes": [...],
                  "offlinepanels": [...]
              },
              ...
          }
    """
    import json
    import cv2
    import numpy as np
    from collections import defaultdict
    import rasterio
    from reportlab.graphics.shapes import Drawing, Rect, Polygon, String
    from reportlab.lib import colors
    from reportlab.graphics import renderPDF

    # --- Load image dimensions ---
    with rasterio.open(tif_path) as src:
        transform = src.transform
        img_h, img_w = src.height, src.width

    # Create a ReportLab Drawing with the original image dimensions.
    drawing = Drawing(img_w, img_h)
    # Add a white background.
    drawing.add(Rect(0, 0, img_w, img_h, fillColor=colors.white))

    # --- Read JSON annotations ---
    with open(annotation_json_path, 'r') as f:
        data = json.load(f)
    
    # Flatten all bounding boxes from the JSON.
    all_bboxes = []
    for item in data:
        if "boundingBox" in item and "boundingBoxes" in item["boundingBox"]:
            all_bboxes.extend(item["boundingBox"]["boundingBoxes"])
    
    # Separate bounding boxes by label (all lowercase).
    bboxes_by_label = defaultdict(list)
    for bb in all_bboxes:
        label = bb["label"].lower()
        left  = int(bb["left"])
        top   = int(bb["top"])
        w     = int(bb["width"])
        h     = int(bb["height"])
        # Create a 4-point contour as a list of tuples.
        contour = [(left, top),
                   (left + w, top),
                   (left + w, top + h),
                   (left, top + h)]
        bboxes_by_label[label].append(contour)
    
    # --- Extract panels and defects ---
    default_panels = bboxes_by_label.get("default_panel", [])
    hotspots      = bboxes_by_label.get("hotspots", [])
    faultydiodes  = bboxes_by_label.get("faultydiodes", [])
    offlinepanels = bboxes_by_label.get("offlinepanels", [])
    
    # --- Build bounding rectangles for each panel contour ---
    # These are in the original image coordinate system (top-left origin).
    panel_bboxes = [cv2.boundingRect(np.array(contour, dtype=np.int32))
                    for contour in default_panels]
    
    # --- Group panels into rows using original y coordinate (smallest y is top) ---
    # Sort by y (ascending) then by x.
    panel_bboxes.sort(key=lambda b: (b[1], b[0]))
    rows = []
    if panel_bboxes:
        current_row = [panel_bboxes[0]]
        ref_y = panel_bboxes[0][1]
        for box in panel_bboxes[1:]:
            # If the difference in top values is small (within half the height of the first panel in the row), same row.
            if abs(box[1] - ref_y) <= current_row[0][3] * 0.5:
                current_row.append(box)
            else:
                rows.append(current_row)
                current_row = [box]
                ref_y = box[1]
        if current_row:
            rows.append(current_row)
    
    # Within each row, sort panels by x (left-to-right).
    # We'll keep a dictionary of the original boxes for defect assignment and then transform for drawing.
    panel_grid_orig = {}
    for row_idx, row in enumerate(rows):
        row_sorted = sorted(row, key=lambda b: b[0])
        for col_idx, box in enumerate(row_sorted):
            # Here, row_idx=0 corresponds to the top row.
            panel_grid_orig[(col_idx+1, row_idx+1)] = box

    # Now transform each panel's bounding box from original coordinates (top-left origin)
    # to drawing coordinates (bottom-left origin). For a box (x, y, w, h) in original coords,
    # the drawing y coordinate becomes: new_y = img_h - y - h.
    panel_grid_draw = {}
    for key, box in panel_grid_orig.items():
        x, y, w, h = box
        new_y = img_h - y - h
        panel_grid_draw[key] = (x, new_y, w, h)
    
    # --- Build final dictionary for panels using the drawing coordinates ---
    panel_defects_dict = {}
    for panel_key, panel_box in panel_grid_draw.items():
        panel_defects_dict[panel_key] = {
            "bbox": panel_box,
            "hotspots": [],
            "faultydiodes": [],
            "offlinepanels": []
        }
    
    # --- Assign each defect to the nearest panel ---
    # For defect assignment, we use the original coordinates.
    for defect_label in ["hotspots", "faultydiodes", "offlinepanels"]:
        for contour in bboxes_by_label.get(defect_label, []):
            # Compute bounding rect for defect.
            x, y, w, h = cv2.boundingRect(np.array(contour, dtype=np.int32))
            defect_center = (x + w/2, y + h/2)
            min_dist = float('inf')
            nearest_panel_key = None
            for key, box in panel_grid_orig.items():
                bx, by, bw, bh = box
                panel_center = (bx + bw/2, by + bh/2)
                dist = np.hypot(defect_center[0] - panel_center[0],
                                defect_center[1] - panel_center[1])
                if dist < min_dist:
                    min_dist = dist
                    nearest_panel_key = key
            if nearest_panel_key is None:
                continue
            # Optionally compute geospatial centroid (if needed)
            panel_center_px = (panel_grid_orig[nearest_panel_key][0] + panel_grid_orig[nearest_panel_key][2] / 2,
                               panel_grid_orig[nearest_panel_key][1] + panel_grid_orig[nearest_panel_key][3] / 2)
            lon, lat = transform * panel_center_px
            panel_defects_dict[nearest_panel_key][defect_label].append({
                "bbox": (x, y, w, h),
                "defect_center_px": defect_center,
                "panel_center_px": panel_center_px,
                "panel_centroid_geospatial": (lon, lat)
            })
    
    # ---------------------
    # Draw defect contours (filled) onto the drawing.
    # Transform the y coordinate from original (top-left) to drawing (bottom-left) by: new_y = img_h - y.
    defect_colors = {
        "hotspots": colors.red,
        "faultydiodes": colors.blue,
        "offlinepanels": colors.yellow
    }
    for defect_label, color in defect_colors.items():
        for contour in bboxes_by_label.get(defect_label, []):
            pts = []
            for (px, py) in contour:
                pts.extend([px, img_h - py])  # transform y coordinate
            drawing.add(Polygon(pts, fillColor=color, strokeColor=None))
    
    # Draw panel outlines (and fill with red if there are defects).
    for panel_key, box in panel_grid_draw.items():
        x, y, w, h = box
        pts = [x, y, x+w, y, x+w, y+h, x, y+h]
        issues_count = (len(panel_defects_dict[panel_key]["hotspots"]) +
                        len(panel_defects_dict[panel_key]["faultydiodes"]) +
                        len(panel_defects_dict[panel_key]["offlinepanels"]))
        if issues_count > 0:
            fill_color = colors.Color(1, 0, 0, alpha=0.5)
            stroke_color = colors.red
        else:
            fill_color = None
            stroke_color = colors.black
        drawing.add(Polygon(pts, fillColor=fill_color, strokeColor=stroke_color, strokeWidth=2))
    
    # Add labels for each panel using the keys as they are.
    for panel_key, box in panel_grid_draw.items():
        x, y, w, h = box
        label_str = f"{panel_key[0]}-{panel_key[1]}"
        drawing.add(String(x, y + h + 5, label_str, fontSize=40, fillColor=colors.black))
    
    # Save the drawing as a PDF.
    image_layer_vector = drawing
    renderPDF.drawToFile(drawing, output_image)
    print(f"Annotated map saved as vector PDF to {output_image}")
    
    return panel_defects_dict, image_layer_vector




def annotate_and_crop_defect_area(
    tif_path,
    panel_defects_dict,
    image_layer_vector,
    default_panel_width=127,
    crop_panel_size=5,  # in panel units
    output_dir="output_annotations",
    scale_factor=0.5
):
    """
    For each panel in `panel_defects_dict`, and for each defect type present in that panel:
      1. Determine the panel's bounding box and calculate a crop region (centered on the panel).
      2. Create a PDF vector layer by copying the provided vector drawing (image_layer_vector)
         and overlaying a blue rectangle that marks the crop area. The resulting PDF is saved as:
             "<defect_type>_(col-row)_layer.pdf"
      3. Annotate a copy of the TIFF image with only that defect type’s contours,
         crop around the panel center, downscale, and save as a JPEG:
             "<defect_type>_(col-row)_cropped.jpg"
         
    Args:
        tif_path (str): Path to the TIFF image.
        panel_defects_dict (dict): Dictionary of panel information and defect lists.
        image_layer_vector (reportlab.graphics.shapes.Drawing):
            A ReportLab Drawing (vector) containing the annotated defect map.
        default_panel_width (int): Default panel width in pixels.
        crop_panel_size (int): Crop size in units of panel widths.
        output_dir (str): Directory to save outputs.
        scale_factor (float): Downscale factor for the cropped JPEG.
        
    Returns:
        None. Files are saved to `output_dir`.
    """


    os.makedirs(output_dir, exist_ok=True)

    # --- Load the original TIFF as an OpenCV image ---
    with rasterio.open(tif_path) as src:
        tif_img = reshape_as_image(src.read()).astype(np.uint8)
        tif_img = cv2.cvtColor(tif_img, cv2.COLOR_RGB2BGR)

    # Color map for defect types.
    color_map = {
        "hotspots":      (0, 0, 255),    # Red in BGR
        "faultydiodes":  (255, 0, 0),    # Blue in BGR
        "offlinepanels": (0, 255, 255)   # Yellow in BGR
    }
    defect_types = ["hotspots", "faultydiodes", "offlinepanels"]

    # Assume the vector drawing dimensions match the original image.
    w_layer = image_layer_vector.width
    h_layer = image_layer_vector.height

    # For each panel, process each defect type.
    for panel_key, panel_info in panel_defects_dict.items():
        bx, by, bw, bh = panel_info["bbox"]
        # Calculate the panel center.
        panel_center_x = bx + bw // 2
        panel_center_y = by + bh // 2

        # Determine the crop area (in pixels).
        half_size = (crop_panel_size * default_panel_width) // 2
        xmin = panel_center_x - half_size
        ymin = panel_center_y - half_size
        xmax = panel_center_x + half_size
        ymax = panel_center_y + half_size

        for defect_type in defect_types:
            defect_list = panel_info[defect_type]
            if not defect_list:
                continue  # Skip if no defects of this type

            # Construct file names based on panel key.
            col_idx, row_idx = panel_key  # Assuming panel_key is a tuple like (col, row)
            layer_filename = f"{defect_type}_({col_idx}-{row_idx})_layer.pdf"
            layer_path = os.path.join(output_dir, layer_filename)

            # --- 1) Build a vector layer with blue crop rectangle ---
            # Make a deep copy of the provided drawing so we don't modify the original.
            vector_copy = copy.deepcopy(image_layer_vector)
            # In ReportLab drawings the origin is at the bottom-left.
            # Our crop region was computed using top-left origin coordinates.
            # Convert the crop rectangle’s y-coordinate:
            rect_y = h_layer - ymax
            crop_width = xmax - xmin
            crop_height = ymax - ymin
            vector_copy.add(
                Rect(xmin, rect_y, crop_width, crop_height,
                     strokeColor=colors.blue, strokeWidth=50, fillColor=None)
            )
            renderPDF.drawToFile(vector_copy, layer_path)

            # --- 2) Annotate the TIFF image with only this defect type ---
            annotated_tif = tif_img.copy()
            for defect_data in defect_list:
                dx, dy, dw, dh = defect_data["bbox"]
                contour = np.array([
                    [dx, dy],
                    [dx + dw, dy],
                    [dx + dw, dy + dh],
                    [dx, dy + dh]
                ], dtype=np.int32)
                cv2.drawContours(annotated_tif, [contour], -1, color_map[defect_type], thickness=3)

            # --- 3) Crop the annotated TIFF image ---
            crop_h = half_size * 2
            crop_w = half_size * 2
            cropped_img = np.ones((crop_h, crop_w, 3), dtype=np.uint8) * 255  # white background

            # Determine the source crop region within the TIFF.
            x1_crop = max(0, xmin)
            y1_crop = max(0, ymin)
            x2_crop = min(tif_img.shape[1], xmax)
            y2_crop = min(tif_img.shape[0], ymax)

            # Determine where to paste in the cropped image (if the crop goes off-image).
            target_x1 = max(0, -xmin)
            target_y1 = max(0, -ymin)
            target_x2 = target_x1 + (x2_crop - x1_crop)
            target_y2 = target_y1 + (y2_crop - y1_crop)

            cropped_img[target_y1:target_y2, target_x1:target_x2] = \
                annotated_tif[y1_crop:y2_crop, x1_crop:x2_crop]

            # Save the cropped image as a JPEG.
            cropped_filename = f"{defect_type}_({col_idx}-{row_idx})_cropped.jpg"
            cropped_path = os.path.join(output_dir, cropped_filename)
            downscaled_img = cv2.resize(cropped_img, None, fx=scale_factor, fy=scale_factor,
                                        interpolation=cv2.INTER_AREA)
            cv2.imwrite(cropped_path, downscaled_img)

    print(f"Finished cropping by defect type. Saved results to '{output_dir}'.")




def annotate_and_crop_defect_area(
    tif_path,
    panel_defects_dict,
    image_layer_vector,
    default_panel_width=127,
    crop_panel_size=5,  # in panel units
    output_dir="output_annotations",
    scale_factor=0.5
):
    """
    For each panel in `panel_defects_dict` (whose "bbox" is in drawing coordinates, i.e.
    (x, y, w, h) with origin at bottom-left), do the following:
      1. Convert the panel bbox back to original coordinates (origin at top-left) to compute
         the correct TIFF crop region.
      2. Compute both the drawing center and the original center.
      3. Create a PDF vector layer by copying the provided vector drawing (image_layer_vector)
         and overlaying a blue rectangle (in drawing coordinates) that marks the crop area.
         The resulting PDF is saved as: "<defect_type>_(col-row)_layer.pdf"
      4. Annotate a copy of the TIFF image with only that defect type’s contours,
         crop around the panel center (using original coordinates), downscale,
         and save as a JPEG: "<defect_type>_(col-row)_cropped.jpg"
         
    Args:
        tif_path (str): Path to the TIFF image.
        panel_defects_dict (dict): Dictionary of panel information and defect lists.
            Each panel's "bbox" is in drawing coordinates (origin bottom-left).
        image_layer_vector (reportlab.graphics.shapes.Drawing):
            A ReportLab Drawing (vector) containing the annotated defect map.
        default_panel_width (int): Default panel width in pixels.
        crop_panel_size (int): Crop size in units of panel widths.
        output_dir (str): Directory to save outputs.
        scale_factor (float): Downscale factor for the cropped JPEG.
        
    Returns:
        None. Files are saved to `output_dir`.
    """
    import os, copy
    import cv2
    import numpy as np
    import rasterio
    from rasterio.plot import reshape_as_image
    from reportlab.graphics.shapes import Rect
    from reportlab.lib import colors
    from reportlab.graphics import renderPDF

    os.makedirs(output_dir, exist_ok=True)

    # --- Load the original TIFF image (which uses a top-left origin) ---
    with rasterio.open(tif_path) as src:
        img_h, img_w = src.height, src.width
        tif_img = reshape_as_image(src.read()).astype(np.uint8)
        tif_img = cv2.cvtColor(tif_img, cv2.COLOR_RGB2BGR)

    # Color mapping for defect types.
    color_map = {
        "hotspots":      (0, 0, 255),    # Red
        "faultydiodes":  (255, 0, 0),    # Blue
        "offlinepanels": (0, 255, 255)   # Yellow
    }
    defect_types = ["hotspots", "faultydiodes", "offlinepanels"]

    # Assume vector drawing dimensions match the TIFF.
    w_layer = image_layer_vector.width
    h_layer = image_layer_vector.height

    # Compute half-size for the crop region.
    half_size = (crop_panel_size * default_panel_width) // 2

    # Process each panel.
    # Here, each panel's bbox is (x, y, w, h) in drawing coordinates (origin bottom-left).
    for panel_key, panel_info in panel_defects_dict.items():
        # Unpack panel bbox from drawing coords.
        x_draw, y_draw, panel_w, panel_h = panel_info["bbox"]

        # Compute panel center in drawing coordinates.
        center_draw_x = x_draw + panel_w // 2
        center_draw_y = y_draw + panel_h // 2

        # Convert the panel bbox back to original coordinates.
        # Original x remains the same.
        # Original y = img_h - y_draw - panel_h.
        x_orig = x_draw
        y_orig = img_h - y_draw - panel_h
        center_orig_x = x_orig + panel_w // 2
        center_orig_y = y_orig + panel_h // 2

        # --- Define crop regions ---
        # For vector overlay (drawing coordinates):
        xmin_draw = center_draw_x - half_size
        ymin_draw = center_draw_y - half_size
        crop_width_draw = 2 * half_size
        crop_height_draw = 2 * half_size

        # For TIFF cropping (original coordinates):
        xmin_orig = center_orig_x - half_size
        ymin_orig = center_orig_y - half_size
        xmax_orig = center_orig_x + half_size
        ymax_orig = center_orig_y + half_size

        for defect_type in defect_types:
            defect_list = panel_info[defect_type]
            if not defect_list:
                continue  # Skip if no defects of this type

            # Construct filenames based on panel key.
            col_idx, row_idx = panel_key  # Assuming panel_key is a tuple (col, row)
            layer_filename = f"{defect_type}_({col_idx}-{row_idx})_layer.pdf"
            layer_path = os.path.join(output_dir, layer_filename)

            # --- 1) Build vector layer with blue crop rectangle ---
            # Deep-copy the provided vector drawing.
            vector_copy = copy.deepcopy(image_layer_vector)
            # Overlay the blue rectangle in drawing coordinates.
            vector_copy.add(
                Rect(xmin_draw, ymin_draw, crop_width_draw, crop_height_draw,
                     strokeColor=colors.blue, strokeWidth=50, fillColor=None)
            )
            renderPDF.drawToFile(vector_copy, layer_path)

            # --- 2) Annotate the TIFF with only this defect type ---
            annotated_tif = tif_img.copy()
            for defect_data in defect_list:
                dx, dy, dw, dh = defect_data["bbox"]
                contour = np.array([
                    [dx, dy],
                    [dx + dw, dy],
                    [dx + dw, dy + dh],
                    [dx, dy + dh]
                ], dtype=np.int32)
                cv2.drawContours(annotated_tif, [contour], -1, color_map[defect_type], thickness=3)

            # --- 3) Crop the annotated TIFF using original coordinates ---
            crop_h_orig = 2 * half_size
            crop_w_orig = 2 * half_size
            cropped_img = np.ones((crop_h_orig, crop_w_orig, 3), dtype=np.uint8) * 255  # white background

            # Determine the region to copy from the TIFF.
            x1_crop = max(0, xmin_orig)
            y1_crop = max(0, ymin_orig)
            x2_crop = min(tif_img.shape[1], xmax_orig)
            y2_crop = min(tif_img.shape[0], ymax_orig)

            # Compute paste offset if the crop goes off-image.
            target_x1 = max(0, -xmin_orig)
            target_y1 = max(0, -ymin_orig)
            target_x2 = target_x1 + (x2_crop - x1_crop)
            target_y2 = target_y1 + (y2_crop - y1_crop)

            if (target_y2 - target_y1) > 0 and (target_x2 - target_x1) > 0:
                cropped_img[target_y1:target_y2, target_x1:target_x2] = \
                    annotated_tif[y1_crop:y2_crop, x1_crop:x2_crop]
            else:
                print(f"Warning: Empty crop region for panel {panel_key}, defect {defect_type}.")
                continue

            # Save the cropped image as JPEG.
            cropped_filename = f"{defect_type}_({col_idx}-{row_idx})_cropped.jpg"
            cropped_path = os.path.join(output_dir, cropped_filename)
            downscaled_img = cv2.resize(cropped_img, None, fx=scale_factor, fy=scale_factor,
                                        interpolation=cv2.INTER_AREA)
            cv2.imwrite(cropped_path, downscaled_img)

    print(f"Finished cropping by defect type. Saved results to '{output_dir}'.")






def process_and_rename_images(
    raw_images_dir,
    output_dir,
    panel_defects_dict,
    alignment='vertical',
    quality = 70
):
    """
    Creates exactly ONE image per panel & defect type.

    For each panel in `panel_defects_dict` and each defect type 
    ("hotspots", "faultydiodes", "offlinepanels"), if there is 
    at least one defect:
       1) Compute the average (lon, lat) of all such defects,
       2) Find the closest image (by lat/lon) in `raw_images_dir`,
       3) If that image is 'south'-oriented (see below), rotate 180°,
       4) Resize to half-size,
       5) Save as "<defect_type>_(col-row).jpg".

    Args:
        raw_images_dir (str): Path to .jpg images with EXIF GPS tags.
        output_dir (str): Where to save the processed images.
        panel_defects_dict (dict): Returned by generate_defect_map, e.g.
            {
              (14, 24): {
                "bbox": (...),
                "hotspots": [
                  {
                    "panel_centroid_geospatial": (lon, lat),
                    ...
                  },
                  ...
                ],
                "faultydiodes": [...],
                "offlinepanels": [...]
              },
              ...
            }
        alignment (str): 'vertical' or 'horizontal' (not actually used here, 
                         just included for API consistency).
    """

    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------------------------------------------------
    # 1) Build metadata for .jpg images
    # ----------------------------------------------------------------------
    def tags_to_decimal(tags, ref):
        """Converts EXIF GPS tags [deg, min, sec] to decimal lat/lon."""
        try:
            deg, minutes, seconds = [
                float(str(x).split('/')[0]) / float(str(x).split('/')[1]) 
                if '/' in str(x) else float(x)
                for x in tags
            ]
        except:
            return None

        decimal = deg + (minutes / 60.0) + (seconds / 3600.0)
        if ref in ['S', 'W']:
            decimal *= -1
        return decimal

    def extract_metadata(img_path):
        """Reads EXIF lat/lon from an image file, returns {'lat':..., 'lon':...}."""
        with open(img_path, 'rb') as f:
            tags = exifread.process_file(f, details=False)

        lat_val = tags.get('GPS GPSLatitude')
        lat_ref = tags.get('GPS GPSLatitudeRef')
        lon_val = tags.get('GPS GPSLongitude')
        lon_ref = tags.get('GPS GPSLongitudeRef')

        if not (lat_val and lat_ref and lon_val and lon_ref):
            return {'lat': None, 'lon': None}

        lat = tags_to_decimal(lat_val.values, lat_ref.values)
        lon = tags_to_decimal(lon_val.values, lon_ref.values)
        return {
            'lat': lat if lat is not None else None,
            'lon': lon if lon is not None else None
        }

    metadata_dict = {}
    for jpg_path in glob.glob(os.path.join(raw_images_dir, '*.jpg')):
        fname = os.path.basename(jpg_path)
        metadata_dict[fname] = extract_metadata(jpg_path)

    # ----------------------------------------------------------------------
    # 2) Determine orientation per file ('south' or 'north' or None)
    # ----------------------------------------------------------------------
    def determine_orientation_by_filename(meta_dict):
        """
        Sorts images by filename alphabetically. For each file (except last),
        compare lat with next:
          - if next < current => 'south'
          - else => 'north'
        Last file => None
        """
        valid_entries = [
            (fname, coords['lat'])
            for fname, coords in meta_dict.items()
            if coords['lat'] is not None
        ]
        # Sort by filename
        valid_entries.sort(key=lambda x: x[0])

        orientation_map = {}
        for i in range(len(valid_entries)):
            current_fname, current_lat = valid_entries[i]
            if i == len(valid_entries) - 1:
                orientation_map[current_fname] = None
            else:
                _, next_lat = valid_entries[i + 1]
                if next_lat < current_lat:
                    orientation_map[current_fname] = 'south'
                else:
                    orientation_map[current_fname] = 'north'
        return orientation_map

    orientation_map = determine_orientation_by_filename(metadata_dict)

    # ----------------------------------------------------------------------
    # 3) Helper: find the closest .jpg to (lon, lat)
    # ----------------------------------------------------------------------
    def find_closest_image(lon, lat):
        """
        Return the filename with minimal distance in lat/lon space 
        to the defect coordinate (lon, lat).
        """
        min_dist = float('inf')
        closest_fname = None
        for fname, coords in metadata_dict.items():
            if coords['lat'] is None or coords['lon'] is None:
                continue
            # coords are (lat, lon). The defect is (lon, lat).
            d = np.hypot(coords['lat'] - lat, coords['lon'] - lon)
            if d < min_dist:
                min_dist = d
                closest_fname = fname
        return closest_fname

    # ----------------------------------------------------------------------
    # 4) For each panel & each defect type, create only ONE image
    # ----------------------------------------------------------------------
    all_defect_types = ["hotspots", "faultydiodes", "offlinepanels"]

    for panel_key, panel_data in panel_defects_dict.items():
        col_idx, row_idx = panel_key  # e.g. (14,24)

        for defect_type in all_defect_types:
            defects = panel_data[defect_type]
            if not defects:
                continue  # no defects of this type => skip

            # -------------------------
            # (A) Compute the average (lon, lat) of all defects of this type
            # -------------------------
            sum_lon, sum_lat = 0.0, 0.0
            for d in defects:
                lon, lat = d["panel_centroid_geospatial"]
                sum_lon += lon
                sum_lat += lat
            avg_lon = sum_lon / len(defects)
            avg_lat = sum_lat / len(defects)

            # -------------------------
            # (B) Find closest .jpg
            # -------------------------
            closest_fname = find_closest_image(avg_lon, avg_lat)
            if not closest_fname:
                print(f"[WARN] No matching image for {defect_type} in panel {panel_key}")
                continue

            src_path = os.path.join(raw_images_dir, closest_fname)
            if not os.path.isfile(src_path):
                print(f"[WARN] File not found: {src_path}")
                continue

            # -------------------------
            # (C) Open, rotate if south, resize
            # -------------------------
            img = Image.open(src_path)
            orientation = orientation_map.get(closest_fname, None)
            if orientation == 'south':
                img = img.rotate(180, expand=True)

            new_w = img.width // 2
            new_h = img.height // 2
            if new_w > 0 and new_h > 0:
                img = img.resize((new_w, new_h))

            # -------------------------
            # (D) Save as <defect_type>_(col-row).jpg
            # -------------------------
            new_name = f"{defect_type}_({col_idx}-{row_idx}).jpg"
            out_path = os.path.join(output_dir, new_name)
            img.save(out_path, quality=quality)
            print(f"[INFO] Saved => {new_name} (panel={panel_key}, type={defect_type})")

    print(f"[DONE] All processed images in '{output_dir}'.")




def svg_to_pdf(images_dir, export_latex=False):
    """
    Converts all SVG files in the provided directory to PDF files using Inkscape.
    The PDF files will have the same base name as the SVG files.
    Optionally, if export_latex is True, Inkscape will be called with --export-latex,
    producing two files: one PDF and one .pdf_tex file.
    After conversion, the original SVG files are deleted.
    
    Args:
        images_dir (str): Path to the directory containing SVG files.
        export_latex (bool): Whether to export using the PDF+LaTeX option.
    
    Returns:
        None.
    """
    # Check if Inkscape is available
    try:
        subprocess.run(["inkscape", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        print("Inkscape is not installed or not in PATH:", e)
        return

    # Loop through SVG files in the directory.
    for filename in os.listdir(images_dir):
        if filename.lower().endswith('.svg'):
            svg_path = os.path.join(images_dir, filename)
            base = os.path.splitext(filename)[0]
            pdf_filename = base + ".pdf"
            pdf_path = os.path.join(images_dir, pdf_filename)
            
            # Build command list: use --export-latex if desired.
            cmd = ["inkscape", "-D", svg_path, "-o", pdf_path]
            if export_latex:
                cmd = ["inkscape", "-D", svg_path, "-o", pdf_path, "--export-latex"]
            try:
                print("Converting:", svg_path)
                subprocess.run(cmd, check=True)
                print(f"Converted {svg_path} to {pdf_path}")
                os.remove(svg_path)
                print(f"Deleted {svg_path}")
            except subprocess.CalledProcessError as e:
                print(f"Error converting {svg_path}: {e}")

