from __future__ import annotations

import re
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np

from app.config import Settings, settings
from app.schemas import BoundingBox, ImageRecognitionResult, PlatePrediction


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

PERSIAN_TO_LATIN_DIGITS = str.maketrans(
    {
        "۰": "0",
        "۱": "1",
        "۲": "2",
        "۳": "3",
        "۴": "4",
        "۵": "5",
        "۶": "6",
        "۷": "7",
        "۸": "8",
        "۹": "9",
        "٠": "0",
        "١": "1",
        "٢": "2",
        "٣": "3",
        "٤": "4",
        "٥": "5",
        "٦": "6",
        "٧": "7",
        "٨": "8",
        "٩": "9",
    }
)
LATIN_TO_PERSIAN_DIGITS = str.maketrans("0123456789", "۰۱۲۳۴۵۶۷۸۹")


def normalize_class_name(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


def normalize_plate_text(text: str, output_persian_digits: bool = False) -> str:
    normalized = str(text).strip().translate(PERSIAN_TO_LATIN_DIGITS)
    normalized = normalized.replace("ي", "ی").replace("ك", "ک")
    normalized = re.sub(r"[\s\-_.:/\\|]+", "", normalized)
    normalized = re.sub(r"[^0-9A-Za-zآ-ی♿]", "", normalized)
    if output_persian_digits:
        normalized = normalized.translate(LATIN_TO_PERSIAN_DIGITS)
    return normalized


def iter_image_files(folder: Path, recursive: bool = False) -> list[Path]:
    iterator: Iterable[Path] = folder.rglob("*") if recursive else folder.glob("*")
    return sorted(
        path.resolve()
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


class LPRService:
    """Thread-safe Iranian LPR service.

    Stage 1: an Ultralytics YOLO model detects plate boxes.
    Stage 2: a local Hezar CRNN model reads each complete plate crop.
    """

    REQUIRED_RECOGNIZER_FILES = (
        Path("model.pt"),
        Path("model_config.yaml"),
        Path("preprocessor") / "image_processor_config.yaml",
    )

    def __init__(self, app_settings: Settings = settings) -> None:
        self.settings = app_settings
        self._detector: Any | None = None
        self._recognizer: Any | None = None
        self._load_error: Exception | None = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        (self.settings.output_dir / "previews").mkdir(parents=True, exist_ok=True)

    @property
    def models_loaded(self) -> bool:
        return self._detector is not None and self._recognizer is not None

    @property
    def model_load_error(self) -> str | None:
        return str(self._load_error) if self._load_error is not None else None

    def recognizer_missing_files(self) -> list[Path]:
        root = self.settings.recognizer_model_dir
        return [root / relative for relative in self.REQUIRED_RECOGNIZER_FILES if not (root / relative).is_file()]

    def ensure_models_loaded(self) -> None:
        if self.models_loaded:
            return
        if self._load_error is not None:
            raise RuntimeError(f"Model loading previously failed: {self._load_error}") from self._load_error

        with self._load_lock:
            if self.models_loaded:
                return
            if self._load_error is not None:
                raise RuntimeError(f"Model loading previously failed: {self._load_error}") from self._load_error

            try:
                detector_path = self.settings.detector_weights.resolve()
                recognizer_dir = self.settings.recognizer_model_dir.resolve()

                if not detector_path.is_file():
                    raise FileNotFoundError(
                        f"Plate detector checkpoint was not found: {detector_path}"
                    )
                if not recognizer_dir.is_dir():
                    raise FileNotFoundError(
                        "The Hezar recognizer must be a local model directory, not a standalone .pt file. "
                        f"Directory was not found: {recognizer_dir}"
                    )

                missing = self.recognizer_missing_files()
                if missing:
                    raise FileNotFoundError(
                        "The local Hezar recognizer directory is incomplete. Missing: "
                        + ", ".join(str(path) for path in missing)
                    )

                try:
                    from ultralytics import YOLO
                except ImportError as exc:  # pragma: no cover
                    raise RuntimeError(
                        "ultralytics is not installed. Run: pip install -r requirements.txt"
                    ) from exc

                try:
                    from hezar.models import Model
                except ImportError as exc:  # pragma: no cover
                    raise RuntimeError(
                        "hezar is not installed. Run: pip install -r requirements.txt"
                    ) from exc

                print(f"[LPR] Loading plate detector: {detector_path}", flush=True)
                detector = YOLO(str(detector_path))

                print(f"[LPR] Loading local Hezar recognizer: {recognizer_dir}", flush=True)
                # load_locally=True is important on Windows. Without it, an invalid or
                # incomplete local path can be interpreted as a Hugging Face repository ID.
                recognizer = Model.load(str(recognizer_dir), load_locally=True)
                recognizer.eval()

                torch_device = self._torch_device()
                recognizer.to(torch_device)

                self._detector = detector
                self._recognizer = recognizer
                print(f"[LPR] Models loaded successfully on {torch_device}", flush=True)
            except Exception as exc:
                self._load_error = exc
                raise RuntimeError(f"Failed to load LPR models: {exc}") from exc

    def _torch_device(self) -> str:
        configured = self.settings.device.strip().lower()
        if configured == "cpu":
            return "cpu"
        if configured.startswith("cuda"):
            return configured
        if configured.isdigit():
            return f"cuda:{configured}"
        return self.settings.device

    def _model_names(self, model: Any) -> dict[int, str]:
        names = getattr(model, "names", {})
        if isinstance(names, list):
            return {index: str(name) for index, name in enumerate(names)}
        return {int(index): str(name) for index, name in dict(names).items()}

    def _plate_class_ids(self) -> set[int]:
        if self.settings.plate_class_ids:
            return set(self.settings.plate_class_ids)

        assert self._detector is not None
        names = self._model_names(self._detector)
        if len(names) == 1:
            return set(names)

        matched = {
            class_id
            for class_id, name in names.items()
            if any(
                token in normalize_class_name(name)
                for token in ("plate", "license_plate", "number_plate", "پلاک")
            )
        }
        if matched:
            return matched

        raise RuntimeError(
            "Could not identify the plate class in the detector. Set "
            "LPR_PLATE_CLASS_IDS to the detector class ID, for example: LPR_PLATE_CLASS_IDS=1"
        )

    def recognize_image_path(
        self,
        image_path: Path,
        preview: bool = False,
        preview_namespace: str | None = None,
        raise_errors: bool = False,
    ) -> ImageRecognitionResult:
        image_path = image_path.expanduser().resolve()
        started = time.perf_counter()

        image = cv2.imread(str(image_path))
        if image is None:
            error = "OpenCV could not read the image."
            if raise_errors:
                raise ValueError(error)
            return ImageRecognitionResult(
                image_path=str(image_path),
                filename=image_path.name,
                plates=[],
                processing_ms=(time.perf_counter() - started) * 1000,
                error=error,
            )

        try:
            predictions, annotated = self.recognize_frame(image)
            preview_path: Path | None = None
            preview_url: str | None = None

            if preview:
                namespace = preview_namespace or uuid.uuid4().hex
                output_folder = self.settings.output_dir / "previews" / namespace
                output_folder.mkdir(parents=True, exist_ok=True)
                preview_path = output_folder / image_path.name
                if not cv2.imwrite(str(preview_path), annotated):
                    raise RuntimeError(f"Could not save preview image: {preview_path}")
                preview_url = f"/previews/{namespace}/{image_path.name}"
                self._show_preview_window(image_path.name, annotated)

            plate_texts = [item.plate or "<unreadable>" for item in predictions]
            printed = ", ".join(plate_texts) if plate_texts else "no plate"
            print(f"[LPR] {image_path.name} -> {printed}", flush=True)

            return ImageRecognitionResult(
                image_path=str(image_path),
                filename=image_path.name,
                plates=predictions,
                processing_ms=round((time.perf_counter() - started) * 1000, 3),
                preview_path=str(preview_path) if preview_path else None,
                preview_url=preview_url,
            )
        except Exception as exc:
            if raise_errors:
                raise
            print(f"[LPR][ERROR] {image_path.name} -> {exc}", flush=True)
            return ImageRecognitionResult(
                image_path=str(image_path),
                filename=image_path.name,
                plates=[],
                processing_ms=round((time.perf_counter() - started) * 1000, 3),
                error=str(exc),
            )

    def recognize_frame(self, frame: np.ndarray) -> tuple[list[PlatePrediction], np.ndarray]:
        self.ensure_models_loaded()
        assert self._detector is not None
        assert self._recognizer is not None

        annotated = frame.copy()
        half = self.settings.use_fp16 and self.settings.device.lower() != "cpu"

        with self._inference_lock:
            detector_result = self._detector.predict(
                source=frame,
                conf=self.settings.detector_confidence,
                iou=self.settings.detector_iou,
                imgsz=self.settings.detector_imgsz,
                device=self.settings.device,
                half=half,
                verbose=False,
            )[0]

            predictions: list[PlatePrediction] = []
            allowed_ids = self._plate_class_ids()
            height, width = frame.shape[:2]

            boxes = getattr(detector_result, "boxes", None)
            if boxes is None:
                return predictions, annotated

            for box in boxes:
                class_id = int(box.cls[0].item())
                if class_id not in allowed_ids:
                    continue

                x1, y1, x2, y2 = [int(round(value)) for value in box.xyxy[0].tolist()]
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(x1 + 1, min(x2, width))
                y2 = max(y1 + 1, min(y2, height))
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                plate, recognizer_confidence = self._recognize_crop(crop)
                detector_confidence = float(box.conf[0].item())
                prediction = PlatePrediction(
                    plate=plate,
                    detector_confidence=round(detector_confidence, 5),
                    recognizer_confidence=round(recognizer_confidence, 5),
                    bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                    characters=list(plate),
                )
                predictions.append(prediction)
                self._draw_prediction(annotated, prediction)

        return self._deduplicate(predictions), annotated

    def _recognize_crop(self, crop: np.ndarray) -> tuple[str, float]:
        assert self._recognizer is not None

        # OpenCV frames are BGR; Hezar's image processor expects normal RGB images.
        rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        outputs = self._recognizer.predict(
            rgb_crop,
            device=self._torch_device(),
            return_scores=True,
        )
        if outputs is None:
            return "", 0.0
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        if not outputs:
            return "", 0.0

        output = outputs[0]
        if isinstance(output, dict):
            raw_text = output.get("text", "")
            raw_score = output.get("score", 0.0)
        else:
            raw_text = getattr(output, "text", str(output) if output is not None else "")
            raw_score = getattr(output, "score", 0.0)

        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            score = 0.0

        plate = normalize_plate_text(
            str(raw_text),
            output_persian_digits=self.settings.output_persian_digits,
        )
        return plate, score

    @staticmethod
    def _deduplicate(predictions: list[PlatePrediction]) -> list[PlatePrediction]:
        best: dict[str, PlatePrediction] = {}
        unreadable: list[PlatePrediction] = []
        for prediction in predictions:
            if not prediction.plate:
                unreadable.append(prediction)
                continue
            previous = best.get(prediction.plate)
            if previous is None or (
                prediction.detector_confidence + prediction.recognizer_confidence
                > previous.detector_confidence + previous.recognizer_confidence
            ):
                best[prediction.plate] = prediction
        return list(best.values()) + unreadable

    @staticmethod
    def _draw_prediction(image: np.ndarray, prediction: PlatePrediction) -> None:
        box = prediction.bbox
        cv2.rectangle(image, (box.x1, box.y1), (box.x2, box.y2), (0, 255, 0), 2)
        # OpenCV's built-in font does not render Persian reliably. The number is
        # still returned and printed correctly; the preview falls back to "plate"
        # when the string includes non-ASCII characters.
        
        label = prediction.plate 
        label = label or "plate"
        cv2.putText(
            image,
            label,
            (box.x1, max(20, box.y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    def _show_preview_window(self, window_name: str, image: np.ndarray) -> None:
        if not self.settings.show_window:
            return
        try:
            cv2.imshow(f"LPR - {window_name}", image)
            cv2.waitKey(1)
        except cv2.error as exc:
            print(
                f"[LPR][PREVIEW] OpenCV window is unavailable; saved preview is still available: {exc}",
                flush=True,
            )
