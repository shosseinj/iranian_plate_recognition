from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from app.config import settings
from app.services.lpr import LPRService


class FakeBox:
    def __init__(self, cls_id: int, xyxy: list[float], confidence: float) -> None:
        self.cls = np.array([cls_id], dtype=np.float32)
        self.xyxy = np.array([xyxy], dtype=np.float32)
        self.conf = np.array([confidence], dtype=np.float32)


class FakeResult:
    def __init__(self, boxes: list[FakeBox]) -> None:
        self.boxes = boxes


class FakeDetector:
    def __init__(self) -> None:
        self.names = {0: "car", 1: "plate"}

    def predict(self, **_: object) -> list[FakeResult]:
        return [FakeResult([FakeBox(1, [10, 10, 150, 60], 0.91)])]


@dataclass
class FakeOCRResult:
    text: str
    score: float


class FakeHezarRecognizer:
    def predict(self, *_: object, **__: object) -> list[FakeOCRResult]:
        return [FakeOCRResult(text="۱۲ ب ۳۴۵۶۷", score=0.94)]


def test_yolo_detector_and_hezar_recognizer_with_preview(tmp_path: Path) -> None:
    app_settings = replace(
        settings,
        device="cpu",
        use_fp16=False,
        output_dir=tmp_path / "output",
    )
    service = LPRService(app_settings)
    service._detector = FakeDetector()
    service._recognizer = FakeHezarRecognizer()

    image_path = tmp_path / "camera.jpg"
    cv2.imwrite(str(image_path), np.zeros((100, 200, 3), dtype=np.uint8))

    result = service.recognize_image_path(image_path, preview=True, preview_namespace="test")

    assert result.error is None
    assert [item.plate for item in result.plates] == ["12ب34567"]
    assert result.plates[0].recognizer_confidence == 0.94
    assert result.preview_path is not None
    assert Path(result.preview_path).is_file()
    assert result.preview_url == "/previews/test/camera.jpg"
