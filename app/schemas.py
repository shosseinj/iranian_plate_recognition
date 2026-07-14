from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class FolderRecognitionRequest(BaseModel):
    folder_path: str = Field(
        default="input_images",
        description="Server-side folder containing images.",
    )
    preview: bool = Field(
        default=False,
        description="When true, annotated images are saved and preview URLs are returned.",
    )
    recursive: bool = False


class WatchStartRequest(FolderRecognitionRequest):
    process_existing: bool = True
    poll_interval_ms: int = Field(default=250, ge=50, le=60_000)


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int


class PlatePrediction(BaseModel):
    plate: str
    detector_confidence: float
    recognizer_confidence: float
    bbox: BoundingBox
    characters: list[str]


class ImageRecognitionResult(BaseModel):
    image_path: str
    filename: str
    plates: list[PlatePrediction]
    processing_ms: float
    preview_path: str | None = None
    preview_url: str | None = None
    error: str | None = None


class FolderRecognitionResponse(BaseModel):
    folder_path: str
    image_count: int
    plate_count: int
    results: list[ImageRecognitionResult]
    elapsed_ms: float

from pydantic import BaseModel, Field


class VideoRecognitionRequest(BaseModel):
    video_path: str = Field(
        ...,
        min_length=1,
        description="Absolute or project-relative path to the MP4 video.",
    )

    preview: bool = Field(
        default=False,
        description="Save annotated preview frames when enabled.",
    )

    frame_interval: int = Field(
        default=1,
        ge=1,
        description="Process every Nth frame. Use 1 to process every frame.",
    )

    max_processed_frames: int | None = Field(
        default=None,
        ge=1,
        description="Optional maximum number of frames to process.",
    )
    
class WatchJobResponse(BaseModel):
    job_id: str
    status: Literal["running", "stopped", "failed"]
    folder_path: str
    preview: bool
    processed_images: int
    detected_plates: int
    started_at: datetime
    updated_at: datetime
    error: str | None = None
    recent_results: list[ImageRecognitionResult] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    detector_weights: str
    recognizer_model_dir: str
    detector_exists: bool
    recognizer_dir_exists: bool
    recognizer_complete: bool
    recognizer_missing_files: list[str]
    models_loaded: bool
    model_load_error: str | None = None


class StopWatchResponse(BaseModel):
    job_id: str
    stopped: bool
