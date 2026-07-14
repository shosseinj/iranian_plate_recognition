from __future__ import annotations

import argparse
import re
from pathlib import Path

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


def natural_sort_key(path: Path) -> list[object]:
    """
    Sort filenames naturally:
    1.png, 2.png, 10.png
    instead of:
    1.png, 10.png, 2.png
    """
    return [
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", path.name)
    ]


def collect_images(
    image_folder: Path,
    recursive: bool = False,
) -> list[Path]:
    iterator = image_folder.rglob("*") if recursive else image_folder.glob("*")

    images = [
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    return sorted(images, key=natural_sort_key)


def read_image_unicode_safe(image_path: Path) -> np.ndarray | None:
    """
    cv2.imread can have problems with some Unicode Windows paths.
    Reading bytes first avoids that issue.
    """
    try:
        image_bytes = np.fromfile(str(image_path), dtype=np.uint8)
        return cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    except Exception:
        return None


def resize_with_letterbox(
    image: np.ndarray,
    target_width: int,
    target_height: int,
) -> np.ndarray:
    """
    Resize without stretching the image.
    Empty regions are filled with black pixels.
    """
    source_height, source_width = image.shape[:2]

    if source_width <= 0 or source_height <= 0:
        raise ValueError("Image has an invalid size")

    scale = min(
        target_width / source_width,
        target_height / source_height,
    )

    resized_width = max(1, int(round(source_width * scale)))
    resized_height = max(1, int(round(source_height * scale)))

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA
        if scale < 1
        else cv2.INTER_LINEAR,
    )

    frame = np.zeros(
        (target_height, target_width, 3),
        dtype=np.uint8,
    )

    x_offset = (target_width - resized_width) // 2
    y_offset = (target_height - resized_height) // 2

    frame[
        y_offset : y_offset + resized_height,
        x_offset : x_offset + resized_width,
    ] = resized

    return frame


def make_even(value: int) -> int:
    """
    Many video codecs require even frame dimensions.
    """
    return value if value % 2 == 0 else value + 1


def create_video_from_images(
    image_folder: Path,
    output_path: Path,
    fps: float = 10.0,
    seconds_per_image: float | None = None,
    width: int | None = None,
    height: int | None = None,
    recursive: bool = False,
) -> None:
    if fps <= 0:
        raise ValueError("fps must be greater than zero")

    if seconds_per_image is not None and seconds_per_image <= 0:
        raise ValueError("seconds_per_image must be greater than zero")

    if not image_folder.is_dir():
        raise FileNotFoundError(
            f"Image folder does not exist: {image_folder.resolve()}"
        )

    image_paths = collect_images(
        image_folder=image_folder,
        recursive=recursive,
    )

    if not image_paths:
        raise FileNotFoundError(
            f"No supported images found in: {image_folder.resolve()}"
        )

    first_image = None
    first_valid_index = None

    for index, image_path in enumerate(image_paths):
        first_image = read_image_unicode_safe(image_path)

        if first_image is not None:
            first_valid_index = index
            break

        print(f"[WARNING] Could not read: {image_path.name}")

    if first_image is None or first_valid_index is None:
        raise RuntimeError("None of the images could be decoded")

    if first_valid_index > 0:
        image_paths = image_paths[first_valid_index:]

    source_height, source_width = first_image.shape[:2]

    output_width = make_even(width or source_width)
    output_height = make_even(height or source_height)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # mp4v is broadly supported by OpenCV on Windows.
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (output_width, output_height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            "Could not initialize the MP4 writer. "
            "Check your OpenCV video codec installation."
        )

    frames_per_image = 1

    if seconds_per_image is not None:
        frames_per_image = max(
            1,
            int(round(fps * seconds_per_image)),
        )

    written_images = 0
    written_frames = 0

    try:
        for index, image_path in enumerate(image_paths, start=1):
            image = read_image_unicode_safe(image_path)

            if image is None:
                print(f"[WARNING] Skipping unreadable image: {image_path.name}")
                continue

            frame = resize_with_letterbox(
                image=image,
                target_width=output_width,
                target_height=output_height,
            )

            for _ in range(frames_per_image):
                writer.write(frame)
                written_frames += 1

            written_images += 1

            print(
                f"[VIDEO] {index}/{len(image_paths)} "
                f"{image_path.name}"
            )

    finally:
        writer.release()

    if written_images == 0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("No valid images were written to the video")

    duration = written_frames / fps

    print()
    print("[SUCCESS] Video created")
    print(f"Output: {output_path.resolve()}")
    print(f"Images: {written_images}")
    print(f"Frames: {written_frames}")
    print(f"FPS: {fps}")
    print(f"Resolution: {output_width}x{output_height}")
    print(f"Duration: {duration:.2f} seconds")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert a folder of images into an MP4 video."
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=Path("../plate/img_plate/VehiclePlates"),
        help="Folder containing input images.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("mp4_folder/cars.mp4"),
        help="Output MP4 path.",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=0.5,
        help="Output video frame rate.",
    )

    parser.add_argument(
        "--seconds-per-image",
        type=float,
        default=None,
        help=(
            "Display each image for this many seconds. "
            "When omitted, each image becomes one video frame."
        ),
    )

    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Output width. Defaults to the first image width.",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=640,
        help="Output height. Defaults to the first image height.",
    )

    parser.add_argument(
        "--recursive",
        default=False,
        action="store_true",
        help="Include images from subfolders.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    create_video_from_images(
        image_folder=args.input,
        output_path=args.output,
        fps=args.fps,
        seconds_per_image=args.seconds_per_image,
        width=args.width,
        height=args.height,
        recursive=args.recursive,
    )


if __name__ == "__main__":
    main()