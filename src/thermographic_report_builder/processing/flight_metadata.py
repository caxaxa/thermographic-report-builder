"""Flight facts the raw frames record: capture time, camera, drone and serial numbers.

Reads only the first 64 KB of a few frames. DJI writes the EXIF and XMP blocks first, ahead of
the radiometric data, so the whole file is not needed.
"""

import re
import statistics
from datetime import datetime
from typing import Callable, Iterable, Optional

from PIL import Image

from ..models.inspection import FlightMetadata
from ..utils.logger import get_logger

logger = get_logger(__name__)

HEAD_BYTES = 65536

_EXIF_IFD = 0x8769
_TAG_MAKE = 0x010F
_TAG_MODEL = 0x0110
_TAG_DATETIME_ORIGINAL = 0x9003
_TAG_BODY_SERIAL = 0xA431

# DJI writes XMP both as attributes and as elements, depending on the model.
_XMP_ATTR = re.compile(rb'drone-dji:(\w+)="([^"]*)"')
_XMP_ELEM = re.compile(rb"<drone-dji:(\w+)>([^<]*)</drone-dji:\1>")


def _segments(data: bytes) -> Iterable[tuple[int, bytes]]:
    """Yield (marker, payload) for the JPEG segments ahead of the scan data."""
    i = 2
    while i + 4 <= len(data) and data[i] == 0xFF:
        marker = data[i + 1]
        length = int.from_bytes(data[i + 2:i + 4], "big")
        yield marker, data[i + 4:i + 2 + length]
        if marker == 0xDA:
            return
        i += 2 + length


def _text(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8", "ignore")
    value = str(value).strip("\x00 ").strip()
    return value or None


def _number(value: Optional[str]) -> Optional[float]:
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


def read_frame_metadata(head: bytes) -> dict:
    """Capture time, make, model, serials and gimbal data from the start of a JPEG file."""
    meta: dict = {}
    if not head.startswith(b"\xff\xd8"):
        return meta
    xmp: dict = {}
    for marker, payload in _segments(head):
        if marker != 0xE1:
            continue
        if payload.startswith(b"Exif\x00\x00"):
            exif = Image.Exif()
            exif.load(payload)
            sub = exif.get_ifd(_EXIF_IFD)
            meta["make"] = _text(exif.get(_TAG_MAKE))
            meta["model"] = _text(exif.get(_TAG_MODEL))
            meta["body_serial"] = _text(sub.get(_TAG_BODY_SERIAL))
            captured = _text(sub.get(_TAG_DATETIME_ORIGINAL))
            if captured:
                try:
                    meta["captured"] = datetime.strptime(captured, "%Y:%m:%d %H:%M:%S")
                except ValueError:
                    logger.debug(f"Unreadable DateTimeOriginal: {captured!r}")
        elif b"<x:xmpmeta" in payload:
            for pattern in (_XMP_ATTR, _XMP_ELEM):
                for key, value in pattern.findall(payload):
                    xmp.setdefault(key.decode(), _text(value))
    meta["camera_serial"] = xmp.get("CameraSerialNumber")
    meta["drone_model"] = xmp.get("DroneModel")
    meta["drone_serial"] = xmp.get("DroneSerialNumber")
    meta["gimbal_pitch"] = _number(xmp.get("GimbalPitchDegree"))
    meta["rangefinder"] = _number(xmp.get("LRFTargetDistance"))
    meta["rangefinder_status"] = xmp.get("LRFStatus")
    return meta


def collect_flight_metadata(keys: list[str], read_head: Callable[[str], bytes]) -> FlightMetadata:
    """Sample the first, middle and last frames; DJI file names follow capture order."""
    if not keys:
        return FlightMetadata()
    ordered = sorted(keys)
    sample = list(dict.fromkeys([ordered[0], ordered[len(ordered) // 2], ordered[-1]]))
    metas = []
    for key in sample:
        try:
            metas.append(read_frame_metadata(read_head(key)))
        except Exception as e:
            logger.warning(f"Could not read the metadata of {key}: {e}")
    if not metas:
        return FlightMetadata(frame_count=len(keys))

    def first(field: str) -> Optional[str]:
        return next((m[field] for m in metas if m.get(field)), None)

    def median(field: str) -> Optional[float]:
        values = [m[field] for m in metas if m.get(field) is not None]
        return statistics.median(values) if values else None

    times = sorted(m["captured"] for m in metas if m.get("captured"))
    flight = FlightMetadata(
        start=times[0] if times else None,
        end=times[-1] if times else None,
        frame_count=len(keys),
        camera_make=first("make"),
        camera_model=first("model"),
        camera_serial=first("camera_serial") or first("body_serial"),
        drone_model=first("drone_model"),
        drone_serial=first("drone_serial"),
        gimbal_pitch_deg=median("gimbal_pitch"),
        rangefinder_distance_m=median("rangefinder"),
    )
    logger.info(
        f"Flight: {flight.start} to {flight.end}, {flight.frame_count} frames, "
        f"camera {flight.camera_make} {flight.camera_model} (S/N {flight.camera_serial})"
    )
    return flight
