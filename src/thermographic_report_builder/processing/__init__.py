"""Image processing modules for defect mapping and annotation."""

from .defect_mapper import DefectMapper
from .annotation import annotate_orthophoto, create_layer_image
from .cropper import crop_defect_regions
from .gps_matcher import GPSMatcher, DefectMatch
from .dxf_generator import create_dxf_layers, create_dxf_from_orthophoto
from .camera_projector import CameraProjector, RawImageMatch, load_reconstruction
from .thermal_extractor import ThermalExtractor, TemperatureReading
from .thermal_annotator import ThermalAnnotator, AnnotatedThermalImage
from .crop_transform import CropTransform

__all__ = [
    "DefectMapper",
    "annotate_orthophoto",
    "create_layer_image",
    "crop_defect_regions",
    "GPSMatcher",
    "DefectMatch",
    "create_dxf_layers",
    "create_dxf_from_orthophoto",
    "CameraProjector",
    "RawImageMatch",
    "load_reconstruction",
    "ThermalExtractor",
    "TemperatureReading",
    "ThermalAnnotator",
    "AnnotatedThermalImage",
    "CropTransform",
]
