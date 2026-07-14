from __future__ import annotations

import argparse
from pathlib import Path

from app.services.lpr import LPRService, iter_image_files


def main() -> int:
    parser = argparse.ArgumentParser(description="Process a folder of images with Iranian LPR.")
    parser.add_argument("--folder", default="input_images", help="Folder containing images")
    parser.add_argument("--preview", action="store_true", help="Save annotated preview images")
    parser.add_argument("--recursive", action="store_true", help="Scan subdirectories")
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    if not folder.is_dir():
        parser.error(f"Folder does not exist: {folder}")

    service = LPRService()
    # Fail once before processing the folder when either model is invalid.
    service.ensure_models_loaded()

    files = iter_image_files(folder, recursive=args.recursive)
    total_plates = 0
    for image_path in files:
        result = service.recognize_image_path(image_path, preview=args.preview)
        total_plates += len([item for item in result.plates if item.plate])

    print(f"Processed images: {len(files)}")
    print(f"Recognized plates: {total_plates}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
