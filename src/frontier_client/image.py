"""Validate text+image inputs before creating official SDK image content."""
from __future__ import annotations

import ipaddress
from pathlib import Path
from urllib.parse import urlsplit

from .errors import ImageInputError

MAX_IMAGE_BYTES = 20 * 1024 * 1024
EXTENSIONS = {".jpg", ".jpeg", ".png"}


def sanitized_image_ref(value: str | Path) -> str:
    """Keep an image reference for export without signed URL query secrets."""
    raw = str(value)
    url = urlsplit(raw)
    return url._replace(query="", fragment="").geturl() if url.scheme in {"https", "http"} else raw


def prepare_image(value: str | Path, sdk_images: object) -> tuple[object, str]:
    """Return (official SDK ImageContent, safe reference); no raw bytes in logs."""
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ImageInputError("Image must be a local PNG/JPEG path or public HTTPS URL.")
    raw = str(value)
    url = urlsplit(raw)
    if url.scheme in {"https", "http"}:
        host = url.hostname or ""
        if url.scheme != "https" or not host or url.username or url.password or host.lower() == "localhost" or host.lower().endswith(".local"):
            raise ImageInputError("Only public HTTPS image URLs without embedded credentials are supported.")
        try:
            addr = ipaddress.ip_address(host)
        except ValueError:
            addr = None  # A public DNS hostname, not an IP literal.
        if addr is not None and not addr.is_global:
            raise ImageInputError("Local or private IP addresses are not valid public image URLs.")
        if Path(url.path).suffix.lower() not in EXTENSIONS:
            raise ImageInputError("Image URL must end in .png, .jpg, or .jpeg.")
        # Avoid persisting signed/private URL query parameters in JSONL logs.
        return sdk_images.from_url(raw), sanitized_image_ref(raw)
    if url.scheme:
        raise ImageInputError("Only local files or public HTTPS image URLs are accepted.")
    path = Path(raw).expanduser()
    if not path.is_file():
        raise ImageInputError(f"Image does not exist: {path}")
    if path.suffix.lower() not in EXTENSIONS:
        raise ImageInputError("Only PNG and JPEG images are supported.")
    if path.stat().st_size > MAX_IMAGE_BYTES:
        raise ImageInputError(f"Image exceeds the {MAX_IMAGE_BYTES // (1024 * 1024)} MB limit.")
    if path.stat().st_size < 4:
        raise ImageInputError("Image file is empty or invalid.")
    with path.open("rb") as stream:
        signature = stream.read(8)
    if path.suffix.lower() == ".png" and signature != b"\x89PNG\r\n\x1a\n":
        raise ImageInputError("File is not a valid PNG.")
    if path.suffix.lower() in {".jpg", ".jpeg"} and not signature.startswith(b"\xff\xd8\xff"):
        raise ImageInputError("File is not a valid JPEG.")
    return sdk_images.from_path(str(path)), str(path)
