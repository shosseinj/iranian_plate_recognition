from pathlib import Path

import cv2
import numpy as np

from app.services.lpr import iter_image_files, normalize_plate_text


def test_normalize_plate_text() -> None:
    assert normalize_plate_text(" ۱۲-ب-۳۴۵۶۷ ") == "12ب34567"
    assert normalize_plate_text("12ب34567", output_persian_digits=True) == "۱۲ب۳۴۵۶۷"
    assert normalize_plate_text("١٢ ب ٣٤٥٦٧") == "12ب34567"


def test_iter_image_files(tmp_path: Path) -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    cv2.imwrite(str(tmp_path / "b.jpg"), image)
    cv2.imwrite(str(tmp_path / "a.png"), image)
    (tmp_path / "ignore.txt").write_text("x", encoding="utf-8")

    files = iter_image_files(tmp_path)
    assert [item.name for item in files] == ["a.png", "b.jpg"]
