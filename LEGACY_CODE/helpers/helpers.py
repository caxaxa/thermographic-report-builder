import cv2
import os
import matplotlib.pyplot as plt

def load_orthophoto(path):
    # Read the tif file
    ortho_img =  cv2.imread(path)

    #ortho_img = cv2.cvtColor(ortho_img, cv2.COLOR_BGR2RGB)
    return ortho_img

def display_image(img, cmap=None, figsize=(10,10)):
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap=cmap)
    plt.axis('off')
    plt.show()
    
def get_single_file_in_directory(directory_path):
    """
    Check if there is only one file in the given directory and return its name.
    If there are multiple files or no files, return False.
    """
    files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]
    
    if len(files) == 1:
        return files[0]
    else:
        return False
def flatten_panel_defects_dict(panel_defects_dict):
    """
    Convert a panel_defects_dict like:
      {
        (1,2): {
          "bbox": (x,y,w,h),
          "hotspots": [ { "panel_centroid_geospatial": (...), ... }, ...],
          "faultydiodes": [...],
          "offlinepanels": [...]
        },
        ...
      }
    Into a flat dict:
      {
        "1-2_hotspots_1": {
          "issue_type": "hotspots",
          "panel_centroid_geospatial": (...),
          ...
        },
        "1-2_hotspots_2": { ... },
        ...
      }
    """
    flattened = {}
    defect_types = ["hotspots", "faultydiodes", "offlinepanels"]

    for (col_idx, row_idx), panel_data in panel_defects_dict.items():
        panel_label = f"{col_idx}-{row_idx}"

        for d_type in defect_types:
            # This is a list of defect objects for that d_type
            defects_list = panel_data.get(d_type, [])
            for i, defect_info in enumerate(defects_list, start=1):
                # Build a unique key
                unique_key = f"{panel_label}_{d_type}_{i}"

                # In your data, each defect_info might have keys:
                #   "panel_centroid_geospatial"
                #   "bbox"
                #   ...
                # But NOT necessarily "issue_type" or "label".
                # We'll add "issue_type" explicitly here:
                flattened[unique_key] = {
                    "issue_type": d_type,
                    "panel_centroid_geospatial": defect_info.get("panel_centroid_geospatial", (None, None))
                }
                # If you want other fields, copy them:
                # flattened[unique_key]["bbox"] = defect_info.get("bbox")
                # etc.
    return flattened

def sort_key(item):
    key, _ = item
    # Extract the local coordinate part (e.g., "1-2") from the key.
    local = key.split("_")[0]
    try:
        col, row = local.split("-")
        # Convert to integers for proper numerical comparison.
        return (int(col), int(row))
    except Exception:
        # If the key doesn't match the expected format, push it to the end.
        return (float('inf'), float('inf'))