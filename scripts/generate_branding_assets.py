"""Generate deterministic lemon favicon assets from one geometric design."""

from __future__ import annotations

import argparse
import math
import struct
import zlib
from pathlib import Path

ROOT = Path(__file__).parents[1]
OUTPUT = ROOT / "docs" / "assets" / "branding"
SIZES = (16, 32, 48, 180, 192, 512)

SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64" role="img">
  <title>Lhmily lemon mark</title>
  <desc>A geometric yellow lemon with a green leaf and dark outline.</desc>
  <path d="M39 19 C43 14 48 11 54 11" fill="none" stroke="#111827" stroke-width="4" stroke-linecap="round"/>
  <path d="M43 15 C48 6 58 7 59 8 C58 17 50 21 43 15 Z" fill="#34d399" stroke="#111827" stroke-width="3" stroke-linejoin="round"/>
  <path d="M9 43 C6 34 10 23 18 17 C26 11 38 12 46 19 C54 26 56 38 51 47 C46 56 35 60 25 57 C17 55 11 50 9 43 Z" fill="#facc15" stroke="#111827" stroke-width="4" stroke-linejoin="round"/>
  <path d="M16 39 C14 31 18 23 25 20" fill="none" stroke="#fef08a" stroke-width="4" stroke-linecap="round" opacity="0.9"/>
  <circle cx="18" cy="46" r="2" fill="#eab308"/>
  <circle cx="41" cy="49" r="2.5" fill="#eab308"/>
</svg>
"""


def ellipse(x: float, y: float, cx: float, cy: float, rx: float, ry: float, angle: float) -> bool:
    radians = math.radians(angle)
    dx, dy = x - cx, y - cy
    rotated_x = dx * math.cos(radians) + dy * math.sin(radians)
    rotated_y = -dx * math.sin(radians) + dy * math.cos(radians)
    return (rotated_x / rx) ** 2 + (rotated_y / ry) ** 2 <= 1


def capsule(x: float, y: float, x1: float, y1: float, x2: float, y2: float, radius: float) -> bool:
    dx, dy = x2 - x1, y2 - y1
    length_squared = dx * dx + dy * dy
    t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / length_squared))
    nearest_x, nearest_y = x1 + t * dx, y1 + t * dy
    return (x - nearest_x) ** 2 + (y - nearest_y) ** 2 <= radius**2


def color_at(x: float, y: float) -> tuple[int, int, int, int]:
    transparent = (0, 0, 0, 0)
    dark = (17, 24, 39, 255)
    yellow = (250, 204, 21, 255)
    yellow_shadow = (234, 179, 8, 255)
    highlight = (254, 240, 138, 230)
    green = (52, 211, 153, 255)

    color = transparent
    if capsule(x, y, 39, 19, 54, 11, 2.2):
        color = dark
    if ellipse(x, y, 50.5, 12.5, 10, 5.5, -28):
        color = dark
    if ellipse(x, y, 50.5, 12.5, 7.5, 3.3, -28):
        color = green
    if ellipse(x, y, 31.5, 36, 25.5, 21.5, 34):
        color = dark
    if ellipse(x, y, 31.5, 36, 21.7, 17.7, 34):
        color = yellow
    if capsule(x, y, 18, 38, 25, 21, 2):
        color = highlight
    if ellipse(x, y, 18, 46, 2, 2, 0) or ellipse(x, y, 41, 49, 2.5, 2.5, 0):
        color = yellow_shadow
    return color


def png_bytes(size: int) -> bytes:
    scale = 64 / size
    samples = 4 if size <= 192 else 2
    rows = []
    for py in range(size):
        row = bytearray([0])
        for px in range(size):
            totals = [0, 0, 0, 0]
            for sy in range(samples):
                for sx in range(samples):
                    x = (px + (sx + 0.5) / samples) * scale
                    y = (py + (sy + 0.5) / samples) * scale
                    rgba = color_at(x, y)
                    for index, value in enumerate(rgba):
                        totals[index] += value
            count = samples * samples
            row.extend(round(value / count) for value in totals)
        rows.append(bytes(row))
    raw = b"".join(rows)

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        )

    header = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", size, size, 8, 6, 0, 0, 0)
    return (
        header + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")
    )


def ico_bytes(images: dict[int, bytes]) -> bytes:
    sizes = sorted(images)
    header = struct.pack("<HHH", 0, 1, len(sizes))
    offset = 6 + 16 * len(sizes)
    entries = []
    payload = []
    for size in sizes:
        data = images[size]
        width = 0 if size == 256 else size
        entries.append(struct.pack("<BBBBHHII", width, width, 0, 0, 1, 32, len(data), offset))
        payload.append(data)
        offset += len(data)
    return header + b"".join(entries) + b"".join(payload)


def expected_assets() -> dict[Path, bytes]:
    images = {size: png_bytes(size) for size in SIZES}
    return {
        OUTPUT / "lemon.svg": SVG.encode(),
        OUTPUT / "favicon.png": images[32],
        OUTPUT / "favicon.ico": ico_bytes({size: images[size] for size in (16, 32, 48)}),
        OUTPUT / "apple-touch-icon.png": images[180],
        OUTPUT / "icon-192.png": images[192],
        OUTPUT / "icon-512.png": images[512],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verify assets without writing")
    args = parser.parse_args()
    failures = []
    for path, expected in expected_assets().items():
        if args.check:
            if not path.exists() or path.read_bytes() != expected:
                failures.append(path.relative_to(ROOT))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(expected)
            print(f"wrote {path.relative_to(ROOT)}")
    if failures:
        print("out-of-date assets:", ", ".join(map(str, failures)))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
