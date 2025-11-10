import pytest
from affine import Affine
from pyproj import CRS, Transformer

from thermographic_report_builder.utils.geospatial import PixelToLatLonConverter


def test_pixel_to_lonlat_passthrough_for_wgs84():
    transform = Affine(0.0001, 0, -47.0, 0, -0.0001, -15.0)
    converter = PixelToLatLonConverter(transform, CRS.from_epsg(4326))

    expected_lon, expected_lat = transform * (100, 200)
    lon, lat = converter.pixel_to_lonlat((100, 200))

    assert lon == pytest.approx(expected_lon)
    assert lat == pytest.approx(expected_lat)


def test_pixel_to_lonlat_reprojects_from_utm():
    transform = Affine(0.1, 0, 500000, 0, -0.1, 7200000)
    converter = PixelToLatLonConverter(transform, CRS.from_epsg(32722))

    transformer = Transformer.from_crs("EPSG:32722", "EPSG:4326", always_xy=True)
    easting, northing = transform * (10, 20)
    expected_lon, expected_lat = transformer.transform(easting, northing)

    lon, lat = converter.pixel_to_lonlat((10, 20))

    assert lon == pytest.approx(expected_lon)
    assert lat == pytest.approx(expected_lat)
