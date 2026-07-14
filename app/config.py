from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value not in {None, ""} else default


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value not in {None, ""} else default


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


@dataclass(frozen=True, slots=True)
class Settings:
    app_name: str = os.getenv("APP_NAME", "Iranian LPR FastAPI")
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = _env_int("PORT", 8000)

    detector_weights: Path = _resolve_path(
        os.getenv("PLATE_DETECTOR_WEIGHTS", "weights/plate_detector.pt")
    )
    recognizer_model_dir: Path = _resolve_path(
        os.getenv(
            "PLATE_RECOGNIZER_DIR",
            os.getenv("PLATE_RECOGNIZER_WEIGHTS", "weights/plate_recognizer"),
        )
    )

    device: str = os.getenv("LPR_DEVICE", "0")
    detector_imgsz: int = _env_int("LPR_DETECTOR_IMGSZ", 640)
    detector_confidence: float = _env_float("LPR_DETECTOR_CONFIDENCE", 0.35)
    detector_iou: float = _env_float("LPR_DETECTOR_IOU", 0.45)
    use_fp16: bool = _env_bool("LPR_USE_FP16", True)

    # Optional comma-separated IDs. When empty, class names containing "plate"
    # are selected. For a one-class detector, that class is selected automatically.
    plate_class_ids: tuple[int, ...] = tuple(
        int(item.strip())
        for item in os.getenv("LPR_PLATE_CLASS_IDS", "").split(",")
        if item.strip()
    )

    output_dir: Path = _resolve_path(os.getenv("LPR_OUTPUT_DIR", "output"))
    default_input_dir: Path = _resolve_path(os.getenv("LPR_INPUT_DIR", "input_images"))
    show_window: bool = _env_bool("LPR_SHOW_WINDOW", False)
    output_persian_digits: bool = _env_bool("LPR_OUTPUT_PERSIAN_DIGITS", False)
    max_recent_watch_results: int = _env_int("LPR_MAX_RECENT_WATCH_RESULTS", 200)


settings = Settings()
