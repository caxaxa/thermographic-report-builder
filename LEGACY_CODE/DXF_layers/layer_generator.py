import numpy as np
import rasterio
import ezdxf
import cv2

class GeoImageProcessor:
    def __init__(self, geotiff_path, dxf_path):
        # Opening the GeoTIFF file and getting its geotransform details.
        self.geotransform = self.open_geotiff_and_get_geotransform(geotiff_path)
        #preparing hotspot set to avoid double counting
        self.used_hotspots = set()
        # Reading the DXF file for drawing and annotating.
        self.dwg = ezdxf.readfile(dxf_path)
        self.msp = self.dwg.modelspace()

    def open_geotiff_and_get_geotransform(self, geotiff_path):
        with rasterio.open(geotiff_path) as dataset:
            # Retrieve the geotransform information
            geotransform = dataset.transform
            return geotransform
    def _convert_to_geo_coordinates(self, contour):
        """Convert contour's image coordinates to geospatial coordinates."""
        return [((pt[0][0]) * self.geotransform[0] + self.geotransform[2],
                (pt[0][1]) * self.geotransform[4] + self.geotransform[5]) for pt in contour]

    def draw_contour(self, contour, layer_name, color=7):
        """Draw the contour on the DXF file."""
        geo_contour = self._convert_to_geo_coordinates(contour)
        geo_contour.append(geo_contour[0])
        self.msp.add_lwpolyline(geo_contour, dxfattribs={'layer': layer_name, 'color': color})

    def draw_and_fill_contour(self, contour, layer_name, color=None):
        """Draw and fill the contour on the DXF file."""
        geo_contour = self._convert_to_geo_coordinates(contour)
        self.draw_contour(contour, layer_name, color)
        hatch = self.msp.add_hatch(dxfattribs={'layer': layer_name, 'color': color})
        hatch.paths.add_polyline_path(geo_contour)
        hatch.set_solid_fill(color=color)

    def annotate_contour(self, contour, label, layer_name):
        """Add annotation (text) to the center of the contour."""
        center_x = sum(pt[0] for pt in contour) / len(contour)
        center_y = sum(pt[1] for pt in contour) / len(contour)
        self.msp.add_text(label, dxfattribs={'layer': layer_name, 'insert': (center_x, center_y), 'height': 1})

    def is_panel_affected(self, panel, hotspots, tracker):
        x, y, w, h = cv2.boundingRect(tracker)
        test_panel = np.array([((pt[0][0] + x), (pt[0][1] + y)) for pt in panel])
        offsets = [(0, 0), (5, 0), (5, 0), (0, 3), (0, 3)]
        for hotspot_idx, hotspot in enumerate(hotspots):
            # Check if hotspot has already been used
            if hotspot_idx in self.used_hotspots:
                continue
            M = cv2.moments(hotspot)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                for dx, dy in offsets:
                    if cv2.pointPolygonTest(test_panel, (cx + dx, cy + dy), False) >= 0:
                        self.used_hotspots.add(hotspot_idx)  # Add this hotspot to the used set
                        return True
        return False


    def detect_and_annotate_panels(self, tracker, tracker_idx, new_mask, hotspots,area_name):
        """Detect individual solar panels within a tracker and annotate them."""
        x, y, w, h = cv2.boundingRect(tracker)
        tracker_roi = new_mask[y:y+h, x:x+w]
        inverted_tracker_roi = cv2.bitwise_not(tracker_roi)
        panels = detect_contours(inverted_tracker_roi)
        panels = sorted(panels, key=lambda c: cv2.boundingRect(c)[1])

        for panel_jdx, panel in enumerate(panels):
            panel_geo_contour = [((pt[0][0] + x) * self.geotransform[0] + self.geotransform[2],
                                  (pt[0][1] + y) * self.geotransform[4] + self.geotransform[5]) for pt in panel]
            label = f"{tracker_idx+1}-{panel_jdx+1}"

            # Check if the panel is affected by hotspots.
            if self.is_panel_affected(panel, hotspots, tracker):
                self.msp.add_lwpolyline(panel_geo_contour, dxfattribs={'layer': f'GRETA - {area_name} - Tracker Mask'})
                self.msp.add_lwpolyline(panel_geo_contour, dxfattribs={'layer' : f'GRETA - {area_name} - Affected Trackers', 'color': 1})
                self.annotate_contour(panel_geo_contour, label, f'GRETA - {area_name} - Solar Panels')
                self.annotate_contour(panel_geo_contour, label, f'GRETA - {area_name} - Affected Trackers')
            else:
                self.annotate_contour(panel_geo_contour, label, f'GRETA - {area_name} - Solar Panels')
                self.msp.add_lwpolyline(panel_geo_contour, dxfattribs={'layer': f'GRETA - {area_name} - Tracker Mask'})
    def save(self, path):
        """Save changes to the DXF file."""
        self.dwg.saveas(path)

def detect_contours(mask):
    """Detect contours from a binary mask."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours



def process_geotiff(ortho_path, dxf_file_path,blue_mask, red_mask, area_name):
    processor = GeoImageProcessor(ortho_path, dxf_file_path)

    # Detect the solar panel trackers from the blue mask (new_mask).
    trackers = detect_contours(blue_mask)
    trackers = sorted(trackers, key=lambda c: cv2.boundingRect(c)[0])

    # Detect the hotspots from the red mask (red_mask).
    hotspots = detect_contours(red_mask)
    hotspots = sorted(hotspots, key=lambda c: cv2.boundingRect(c)[0])
    # Draw and fill the detected hotspot contours on the DXF.
    print('Creating  Hotspots Location Layer')
    for hotspot in hotspots:
        processor.draw_and_fill_contour(hotspot, f'GRETA - {area_name} - Hotspots', 1)

    # will be shown in a new mask called 'GRETA - Affected Trackers'.
    print('Creating  Tracker Mask Layer')
    for idx, tracker in enumerate(trackers):
        processor.draw_contour(tracker, f'GRETA - {area_name} - Tracker Mask')
        processor.detect_and_annotate_panels(tracker, idx, blue_mask, hotspots, area_name)

    # Save the modified DXF file.
    processor.save(dxf_file_path)

    return print('Additional layers attached to DXF file.')
    