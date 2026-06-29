#!/usr/bin/env python3
"""
Conical mirror wide-angle image/video unwrapper module.

Pipeline (OpenCV-optimized, one-pass by default):
- Build one inverse remap from the requested square top-view directly to the
  original wide-angle source frame (build_combined_maps).
- Sample via cv2.remap — faster and sharper than the old two-stage numpy path.

The old two-stage numpy path (correct_lens + unwrap_cone + sample_bilinear)
is preserved for debug/tune utilities and can be forced via use_opencv=False.

Usage as module:
    from plane import unwrap_image, debug_parameters

    unwrap_image("input.jpg", cx=575, cy=457, outer_r=412)
    unwrap_image("input.jpg", cx=575, cy=457, outer_r=412, use_opencv=False)  # old path
    debug_parameters("input.jpg", cx=575, cy=457, outer_r=412)  # interactive tuning
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from typing import Tuple, Optional, Union
import numpy as np
from PIL import Image, ImageOps


try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


DEFAULTS = {
    "cx": 1203,
    "cy": 457.0,
    "outer_r": 412.0,
    "rotation_deg": -2.0,
    "top_size": 900,
    "field_scale": 0.70,
    "lens_deg": -81.86,
    "cone_power": 2.245,
}

JPEG_EXTENSIONS = {".jpg", ".jpeg", ".jpe"}
VIDEO_EXTENSIONS = {".mp4", ".m4v", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".jpe", ".bmp", ".tif", ".tiff", ".webp"}


# ============================================================================
# Argument / path helpers
# ============================================================================

def parse_rgb(value: str) -> Tuple[int, int, int]:
    """Parse an RGB triplet like '0,0,0'."""
    parts = [p.strip() for p in value.split(",")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("RGB must be formatted as R,G,B")
    try:
        rgb = tuple(int(p) for p in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("RGB values must be integers") from exc
    if any(v < 0 or v > 255 for v in rgb):
        raise argparse.ArgumentTypeError("RGB values must be in the 0..255 range")
    return rgb  # type: ignore[return-value]


def is_video_path(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_EXTENSIONS


def is_image_path(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


# ============================================================================
# OpenCV helpers (ported from unwrapper_opencv.py)
# ============================================================================

def rgb_to_bgr(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    return (rgb[2], rgb[1], rgb[0])


def border_value_for_channels(rgb: Tuple[int, int, int], channels: int) -> Tuple[int, ...]:
    bgr = rgb_to_bgr(rgb)
    if channels == 1:
        gray = int(round(0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2]))
        return (gray,)
    if channels == 2:
        gray = int(round(0.114 * bgr[0] + 0.587 * bgr[1] + 0.299 * bgr[2]))
        return (gray, 255)
    if channels == 4:
        return (*bgr, 255)
    return bgr


def ensure_video_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def composite_alpha_to_bgr(image: np.ndarray, background_rgb: Tuple[int, int, int]) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 4:
        return ensure_video_bgr(image)
    alpha = image[:, :, 3:4].astype(np.float32) / 255.0
    bgr = image[:, :, :3].astype(np.float32)
    bg = np.array(rgb_to_bgr(background_rgb), dtype=np.float32).reshape(1, 1, 3)
    out = bgr * alpha + bg * (1.0 - alpha)
    return np.clip(out, 0, 255).astype(np.uint8)


def remap_frame(
    frame: np.ndarray,
    map_x: np.ndarray,
    map_y: np.ndarray,
    background_rgb: Tuple[int, int, int],
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    channels = 1 if frame.ndim == 2 else frame.shape[2]
    return cv2.remap(
        frame,
        map_x,
        map_y,
        interpolation=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value_for_channels(background_rgb, channels),
    )


# ============================================================================
# Core math (shared between numpy and OpenCV paths)
# ============================================================================

def lens_distorted_radius(ru_norm: np.ndarray, lens_deg: float) -> np.ndarray:
    """Map corrected normalized radius to source distorted normalized radius."""
    r = np.maximum(ru_norm, 0.0)
    if abs(lens_deg) < 1e-9:
        return r
    angle = abs(lens_deg) * math.pi / 180.0
    limited = min(angle, math.pi / 2.0 - 0.01)
    if lens_deg > 0:
        return np.tan(r * limited) / math.tan(limited)
    return np.arctan(r * math.tan(limited)) / limited


# ============================================================================
# Old numpy two-stage path (kept for debug/tune and optional use)
# ============================================================================

# ============================================================================
# OpenCV remap maps
# ============================================================================

def build_lens_maps(
    height: int,
    width: int,
    cx: float,
    cy: float,
    outer_r: float,
    lens_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse map for the optional wide-angle corrected intermediate image."""
    x_coords = np.arange(width, dtype=np.float32)
    y_coords = np.arange(height, dtype=np.float32)
    xx, yy = np.meshgrid(x_coords, y_coords)
    dx = xx - cx
    dy = yy - cy
    ru = np.sqrt(dx * dx + dy * dy)
    angle = np.arctan2(dy, dx)
    ru_norm = ru / outer_r
    rd = lens_distorted_radius(ru_norm, lens_deg) * outer_r
    map_x = cx + rd * np.cos(angle)
    map_y = cy + rd * np.sin(angle)
    invalid = (
        (map_x < 0.0) |
        (map_y < 0.0) |
        (map_x > width - 1.0) |
        (map_y > height - 1.0)
    )
    map_x = np.where(invalid, -1.0, map_x).astype(np.float32, copy=False)
    map_y = np.where(invalid, -1.0, map_y).astype(np.float32, copy=False)
    return map_x, map_y


def build_unwrap_maps(
    top_size: int,
    cx: float,
    cy: float,
    outer_r: float,
    rotation_deg: float,
    field_scale: float,
    cone_power: float,
    source_width: int | None = None,
    source_height: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse map from square top-view pixels to the lens-corrected frame."""
    size = int(round(top_size))
    center = (size - 1.0) / 2.0
    rotation = math.radians(rotation_deg)
    x_coords = np.arange(size, dtype=np.float32)
    y_coords = np.arange(size, dtype=np.float32)
    xx, yy = np.meshgrid(x_coords, y_coords)
    px = ((xx - center) / center) * field_scale
    py = ((yy - center) / center) * field_scale
    r_plane = np.sqrt(px * px + py * py)
    theta = np.arctan2(py, px) + rotation
    r_corrected = np.power(r_plane, cone_power) * outer_r
    map_x = cx + r_corrected * np.cos(theta)
    map_y = cy + r_corrected * np.sin(theta)
    invalid = r_plane > 1.0
    if source_width is not None and source_height is not None:
        invalid |= (
            (map_x < 0.0) |
            (map_y < 0.0) |
            (map_x > source_width - 1.0) |
            (map_y > source_height - 1.0)
        )
    map_x = np.where(invalid, -1.0, map_x).astype(np.float32, copy=False)
    map_y = np.where(invalid, -1.0, map_y).astype(np.float32, copy=False)
    return map_x, map_y


def build_combined_maps(
    top_size: int,
    source_width: int,
    source_height: int,
    cx: float,
    cy: float,
    outer_r: float,
    rotation_deg: float,
    field_scale: float,
    lens_deg: float,
    cone_power: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    One-pass inverse map from square top-view pixels directly to the original frame.
    Combines lens correction and cone unwrap analytically — sharper and faster
    than the two-stage path.
    """
    size = int(round(top_size))
    center = (size - 1.0) / 2.0
    rotation = math.radians(rotation_deg)
    x_coords = np.arange(size, dtype=np.float32)
    y_coords = np.arange(size, dtype=np.float32)
    xx, yy = np.meshgrid(x_coords, y_coords)
    px = ((xx - center) / center) * field_scale
    py = ((yy - center) / center) * field_scale
    r_plane = np.sqrt(px * px + py * py)
    theta = np.arctan2(py, px) + rotation
    r_corrected = np.power(r_plane, cone_power) * outer_r
    r_distorted = lens_distorted_radius(r_corrected / outer_r, lens_deg) * outer_r
    map_x = cx + r_distorted * np.cos(theta)
    map_y = cy + r_distorted * np.sin(theta)
    invalid = (
        (r_plane > 1.0) |
        (map_x < 0.0) |
        (map_y < 0.0) |
        (map_x > source_width - 1.0) |
        (map_y > source_height - 1.0)
    )
    map_x = np.where(invalid, -1.0, map_x).astype(np.float32, copy=False)
    map_y = np.where(invalid, -1.0, map_y).astype(np.float32, copy=False)
    return map_x, map_y


# ============================================================================
# PIL image I/O (kept for the old numpy path and save_image)
# ============================================================================
def save_image(image: Image.Image, path: Path, background: Tuple[int, int, int]) -> None:
    """Save RGBA images, compositing to RGB only for formats without alpha."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in JPEG_EXTENSIONS:
        background_image = Image.new("RGBA", image.size, (*background, 255))
        image = Image.alpha_composite(background_image, image).convert("RGB")
    image.save(path)


def save_cv_image(path: Path, image: np.ndarray, background_rgb: Tuple[int, int, int]) -> None:
    """Save an OpenCV image (BGR or BGRA) to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    output = image
    if suffix in JPEG_EXTENSIONS:
        output = composite_alpha_to_bgr(image, background_rgb)
    if not cv2.imwrite(str(path), output):
        raise RuntimeError(f"Cannot save output image to {path}")


# ============================================================================
# High-level API
# ============================================================================
def unwrap_image(
        input_path: Union[str, Path],
        cx: float = DEFAULTS["cx"],
        cy: float = DEFAULTS["cy"],
        outer_r: float = DEFAULTS["outer_r"],
        lens_deg: float = DEFAULTS["lens_deg"],
        cone_power: float = DEFAULTS["cone_power"],
        rotation_deg: float = DEFAULTS["rotation_deg"],
        top_size: int = DEFAULTS["top_size"],
        field_scale: float = DEFAULTS["field_scale"],
        output_dir: Union[str, Path] = "Output",
        background: Tuple[int, int, int] = (0, 0, 0),
        chunk_rows: int = 128,
        save_lens_corrected: bool = False,
        use_opencv: bool = True,
        cubic: bool = False,
    ) -> Path:
    """
    Unwrap a conical mirror image to a top-down view.

    By default uses the optimized one-pass OpenCV path (build_combined_maps +
    cv2.remap). Pass use_opencv=False to fall back to the old two-stage numpy
    path (correct_lens + unwrap_cone + sample_bilinear).
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    interpolation = cv2.INTER_CUBIC if cubic else cv2.INTER_LINEAR

    source = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if source is None:
        raise RuntimeError(f"Cannot open input image: {input_path}")
    height, width = source.shape[:2]

    if save_lens_corrected:
        lens_map_x, lens_map_y = build_lens_maps(
            height=height, width=width,
            cx=cx, cy=cy, outer_r=outer_r, lens_deg=lens_deg,
        )
        lens_corrected = remap_frame(
            source, lens_map_x, lens_map_y, background, interpolation,
        )
        lens_path = output_dir / f"{input_path.stem}_lens_corrected{input_path.suffix}"
        save_cv_image(lens_path, lens_corrected, background)

    combined_map_x, combined_map_y = build_combined_maps(
        top_size=top_size,
        source_width=width,
        source_height=height,
        cx=cx, cy=cy, outer_r=outer_r,
        rotation_deg=rotation_deg, field_scale=field_scale,
        lens_deg=lens_deg, cone_power=cone_power,
    )
    top_view = remap_frame(source, combined_map_x, combined_map_y, background, interpolation)
    output_path = output_dir / f"{input_path.stem}_unwrapped{input_path.suffix}"
    save_cv_image(output_path, top_view, background)

    return output_path


# ============================================================================
# Debug / Interactive Tuning
# ============================================================================

def debug_parameters(
        input_path: Union[str, Path],
        cx: float = DEFAULTS["cx"], cy: float = DEFAULTS["cy"],
        outer_r: float = DEFAULTS["outer_r"], lens_deg: float = DEFAULTS["lens_deg"],
        cone_power: float = DEFAULTS["cone_power"], rotation_deg: float = DEFAULTS["rotation_deg"],
        top_size: int = DEFAULTS["top_size"], field_scale: float = DEFAULTS["field_scale"],
        background: Tuple[int, int, int] = (0, 0, 0),
    ) -> None:
    """Interactive parameter tuning via matplotlib sliders."""
    if not (HAS_MATPLOTLIB and HAS_CV2):
        raise RuntimeError("matplotlib and opencv are required for debug mode.")
    from matplotlib.widgets import Slider

    src = cv2.imread(str(input_path), cv2.IMREAD_UNCHANGED)
    if src is None:
        raise RuntimeError(f"Cannot open: {input_path}")
    h, w = src.shape[:2]

    fig, ax = plt.subplots(figsize=(7, 7))
    plt.subplots_adjust(bottom=0.30, left=0.15, right=0.95, top=0.95)
    ax.set_axis_off()

    p = {"cx": cx, "cy": cy, "outer_r": outer_r, "rotation_deg": rotation_deg,
         "field_scale": field_scale, "lens_deg": lens_deg, "cone_power": cone_power, "top_size": top_size}

    def render():
        mx, my = build_combined_maps(
            top_size=int(p["top_size"]), source_width=w, source_height=h,
            cx=p["cx"], cy=p["cy"], outer_r=p["outer_r"], rotation_deg=p["rotation_deg"],
            field_scale=p["field_scale"], lens_deg=p["lens_deg"], cone_power=p["cone_power"])
        out = remap_frame(src, mx, my, background)
        if out.ndim == 3 and out.shape[2] >= 3:
            out = cv2.cvtColor(out[:, :, :3], cv2.COLOR_BGR2RGB)
        return out

    im = ax.imshow(render())
    defs = [
        ("cx", 0, w, 0.26), ("cy", 0, h, 0.23), ("outer_r", 10, max(w, h), 0.20),
        ("rotation_deg", -180, 180, 0.17), ("field_scale", 0.1, 2.0, 0.14),
        ("lens_deg", -90, 90, 0.11), ("cone_power", 0.1, 5.0, 0.08), ("top_size", 100, 2000, 0.05)
    ]
    
    sliders = []
    for name, vmin, vmax, y in defs:
        ax_s = plt.axes([0.25, y, 0.60, 0.02])
        s = Slider(ax_s, name, vmin, vmax, valinit=p[name])
        def make_update(n):
            def update(val):
                p[n] = val
                im.set_data(render())
                fig.canvas.draw_idle()
            return update
        s.on_changed(make_update(name))
        sliders.append(s)

    plt.show()


# ============================================================================
# CLI interface
# ============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert a wide-angle conical-reflector image or video into a square top-down view.",
    )
    parser.add_argument("-i", "--input", required=True, type=Path, help="Source image or video file")
    parser.add_argument("-o", "--output", required=True, type=Path, help="Output image or video file")
    parser.add_argument("--cx", type=float, default=DEFAULTS["cx"], help="Mirror center X")
    parser.add_argument("--cy", type=float, default=DEFAULTS["cy"], help="Mirror center Y")
    parser.add_argument("--outer-r", type=float, default=DEFAULTS["outer_r"], help="Outer mirror radius")
    parser.add_argument("--rotation-deg", type=float, default=DEFAULTS["rotation_deg"], help="Top-view rotation in degrees")
    parser.add_argument("--top-size", type=int, default=DEFAULTS["top_size"], help="Output square size in pixels")
    parser.add_argument("--field-scale", type=float, default=DEFAULTS["field_scale"], help="Top-view field scale")
    parser.add_argument("--lens-deg", type=float, default=DEFAULTS["lens_deg"], help="Wide-angle correction in degrees")
    parser.add_argument("--cone-power", type=float, default=DEFAULTS["cone_power"], help="Cone radial correction power")
    parser.add_argument("--lens-output", type=Path, default=None, help="Optional intermediate wide-angle-corrected image/video")
    parser.add_argument("--background", type=parse_rgb, default=(0, 0, 0), help="Background R,G,B used outside the sampled area")
    parser.add_argument("--chunk-rows", type=int, default=128, help="Rows processed at once for numpy image mode")
    parser.add_argument("--video-codec", default="mp4v", help="FourCC codec for video output")
    parser.add_argument("--video-fps", type=float, default=None, help="Optional output FPS override")
    parser.add_argument("--cubic", action="store_true", help="Use cv2.INTER_CUBIC instead of cv2.INTER_LINEAR")
    parser.add_argument("--numpy", action="store_true", help="Force the old two-stage numpy path for images")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if not args.input.exists():
        raise ValueError("input file does not exist")
    if args.outer_r <= 0:
        raise ValueError("outer-r must be greater than zero")
    if args.top_size < 1:
        raise ValueError("top-size must be greater than zero")
    if args.field_scale <= 0:
        raise ValueError("field-scale must be greater than zero")
    if args.cone_power <= 0:
        raise ValueError("cone-power must be greater than zero")
    if args.chunk_rows < 1:
        raise ValueError("chunk-rows must be greater than zero")
    if len(args.video_codec) != 4:
        raise ValueError("video-codec must contain exactly 4 characters")
    if args.video_fps is not None and args.video_fps <= 0:
        raise ValueError("video-fps must be greater than zero")

    input_is_video = is_video_path(args.input)
    output_is_video = is_video_path(args.output)

    if input_is_video != output_is_video:
        raise ValueError("input and output must both be images or both be videos")

    if not input_is_video and not is_image_path(args.input):
        raise ValueError("unsupported input image extension")
    if not output_is_video and not is_image_path(args.output):
        raise ValueError("unsupported output image extension")

    if args.lens_output is not None:
        if input_is_video != is_video_path(args.lens_output):
            raise ValueError("lens-output must have the same media type as input")
        if not input_is_video and not is_image_path(args.lens_output):
            raise ValueError("unsupported lens-output image extension")


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    if is_video_path(args.input):
        run_video(args)
    else:
        unwrap_image(
            input_path=args.input,
            cx=args.cx,
            cy=args.cy,
            outer_r=args.outer_r,
            lens_deg=args.lens_deg,
            cone_power=args.cone_power,
            rotation_deg=args.rotation_deg,
            top_size=args.top_size,
            field_scale=args.field_scale,
            output_dir=args.output.parent if args.output.parent != Path('.') else Path("Output"),
            background=args.background,
            chunk_rows=args.chunk_rows,
            save_lens_corrected=args.lens_output is not None,
            use_opencv=not args.numpy,
            cubic=args.cubic,
        )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run(args)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    # debug_parameters("Images/New1.jpg", cx=DEFAULTS["cx"], cy=DEFAULTS["cy"], outer_r=DEFAULTS["outer_r"])  # interactive tuning
    run()
