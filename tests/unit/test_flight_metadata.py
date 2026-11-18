"""Tests for flight_metadata — EXIF/XMP from the first bytes of DJI frames."""
import io
from datetime import datetime

from PIL import Image

from thermographic_report_builder.processing.flight_metadata import (
    HEAD_BYTES,
    collect_flight_metadata,
    read_frame_metadata,
)


def _dji_frame(captured="2026:11:10 11:02:30", model="M30T", body_serial="BODY123", xmp_as_elements=False):
    """A small JPEG laid out like a DJI R-JPEG: EXIF APP1, then XMP APP1, then image data."""
    exif = Image.Exif()
    exif[0x010F] = "DJI"
    exif[0x0110] = model
    sub = exif.get_ifd(0x8769)
    sub[0x9003] = captured
    sub[0xA431] = body_serial
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), "gray").save(buf, "JPEG", exif=exif.tobytes())
    data = buf.getvalue()
    if xmp_as_elements:
        fields = (
            b"<drone-dji:CameraSerialNumber>CAM456</drone-dji:CameraSerialNumber>"
            b"<drone-dji:GimbalPitchDegree>-89.90</drone-dji:GimbalPitchDegree>"
        )
    else:
        fields = (
            b'drone-dji:CameraSerialNumber="CAM456" drone-dji:DroneModel="M30T" '
            b'drone-dji:DroneSerialNumber="DRONE789" drone-dji:GimbalPitchDegree="-89.90" '
            b'drone-dji:LRFTargetDistance="24.6"'
        )
    xmp = b"http://ns.adobe.com/xap/1.0/\x00<x:xmpmeta xmlns:x='adobe:ns:meta/'><rdf:Description " + fields + b"/></x:xmpmeta>"
    segment = b"\xff\xe1" + (len(xmp) + 2).to_bytes(2, "big") + xmp
    # Insert the XMP segment right after the EXIF segment, as DJI does
    cut = _exif_end(data)
    return data[:cut] + segment + data[cut:]


def _exif_end(data):
    at = data.find(b"\xff\xe1")
    return at + 2 + int.from_bytes(data[at + 2:at + 4], "big")


class TestReadFrameMetadata:

    def test_reads_exif_and_xmp(self):
        meta = read_frame_metadata(_dji_frame()[:HEAD_BYTES])
        assert meta["make"] == "DJI"
        assert meta["model"] == "M30T"
        assert meta["captured"] == datetime(2026, 11, 10, 11, 2, 30)
        assert meta["camera_serial"] == "CAM456"
        assert meta["drone_model"] == "M30T"
        assert meta["drone_serial"] == "DRONE789"
        assert meta["gimbal_pitch"] == -89.9
        assert meta["rangefinder"] == 24.6

    def test_reads_xmp_written_as_elements(self):
        meta = read_frame_metadata(_dji_frame(xmp_as_elements=True))
        assert meta["camera_serial"] == "CAM456"
        assert meta["gimbal_pitch"] == -89.9

    def test_truncated_head_still_reads_exif(self):
        frame = _dji_frame()
        meta = read_frame_metadata(frame[:_exif_end(frame) + 10])
        assert meta["model"] == "M30T"
        assert meta.get("camera_serial") is None

    def test_not_a_jpeg(self):
        assert read_frame_metadata(b"not a jpeg") == {}


class TestCollectFlightMetadata:

    def test_spans_first_to_last_frame(self):
        frames = {
            "p/images/DJI_0001_T.JPG": _dji_frame("2026:11:10 11:02:30"),
            "p/images/DJI_0002_T.JPG": _dji_frame("2026:11:10 11:10:00"),
            "p/images/DJI_0003_T.JPG": _dji_frame("2026:11:10 11:19:05"),
        }
        flight = collect_flight_metadata(list(frames)[::-1], lambda key: frames[key][:HEAD_BYTES])
        assert flight.start == datetime(2026, 11, 10, 11, 2, 30)
        assert flight.end == datetime(2026, 11, 10, 11, 19, 5)
        assert flight.frame_count == 3
        assert flight.camera_model == "M30T"
        assert flight.camera_serial == "CAM456"
        assert flight.midpoint == datetime(2026, 11, 10, 11, 10, 47, 500000)

    def test_falls_back_to_body_serial(self):
        frame = _dji_frame(xmp_as_elements=True)
        frame = frame.replace(b"CameraSerialNumber", b"OtherSerialNumberX")
        flight = collect_flight_metadata(["a.JPG"], lambda key: frame)
        assert flight.camera_serial == "BODY123"

    def test_unreadable_frames_keep_the_count(self):
        def boom(key):
            raise OSError("no access")
        flight = collect_flight_metadata(["a.JPG", "b.JPG"], boom)
        assert flight.frame_count == 2
        assert flight.start is None

    def test_no_frames(self):
        assert collect_flight_metadata([], lambda key: b"").frame_count == 0
