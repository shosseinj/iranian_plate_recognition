from __future__ import annotations

import shutil
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.schemas import (
    FolderRecognitionRequest,
    FolderRecognitionResponse,
    VideoRecognitionRequest,
    HealthResponse,
    ImageRecognitionResult,
    StopWatchResponse,
    WatchJobResponse,
    WatchStartRequest,
)
from app.services.lpr import IMAGE_EXTENSIONS, LPRService, iter_image_files
from app.services.watcher import FolderWatchManager


lpr_service = LPRService(settings)
watch_manager = FolderWatchManager(lpr_service, settings)

app = FastAPI(
    title=settings.app_name,
    version="1.1.0",
    description=(
        "Backend-only Iranian license plate recognition service. "
        "It processes image uploads or server-side image folders and can watch a folder in real time."
    ),
)

preview_root = settings.output_dir / "previews"
preview_root.mkdir(parents=True, exist_ok=True)
app.mount("/previews", StaticFiles(directory=str(preview_root)), name="previews")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    detector_exists = settings.detector_weights.is_file()
    recognizer_dir_exists = settings.recognizer_model_dir.is_dir()
    missing_files = lpr_service.recognizer_missing_files() if recognizer_dir_exists else []
    recognizer_complete = recognizer_dir_exists and not missing_files
    status_value = "ready" if detector_exists and recognizer_complete else "missing_weights"
    if lpr_service.model_load_error:
        status_value = "model_error"
    return HealthResponse(
        status=status_value,
        detector_weights=str(settings.detector_weights),
        recognizer_model_dir=str(settings.recognizer_model_dir),
        detector_exists=detector_exists,
        recognizer_dir_exists=recognizer_dir_exists,
        recognizer_complete=recognizer_complete,
        recognizer_missing_files=[str(path) for path in missing_files],
        models_loaded=lpr_service.models_loaded,
        model_load_error=lpr_service.model_load_error,
    )


@app.post("/api/v1/lpr/image", response_model=ImageRecognitionResult)
def recognize_uploaded_image(
    file: UploadFile = File(...),
    preview: bool = False,
) -> ImageRecognitionResult:
    suffix = Path(file.filename or "image.jpg").suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported image extension: {suffix}",
        )

    temporary_dir = Path(tempfile.mkdtemp(prefix="iranian-lpr-"))
    image_path = temporary_dir / f"{uuid.uuid4().hex}{suffix}"
    try:
        with image_path.open("wb") as target:
            shutil.copyfileobj(file.file, target)
        try:
            result = lpr_service.recognize_image_path(
                image_path,
                preview=preview,
                raise_errors=True,
            )
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        # Keep the original user-facing filename in the response.
        result.filename = file.filename or image_path.name
        result.image_path = result.filename
        return result
    finally:
        shutil.rmtree(temporary_dir, ignore_errors=True)


@app.post("/api/v1/lpr/folder", response_model=FolderRecognitionResponse)
def recognize_folder(request: FolderRecognitionRequest) -> FolderRecognitionResponse:
    folder = Path(request.folder_path).expanduser()
    if not folder.is_absolute():
        folder = (settings.default_input_dir.parent / folder).resolve()
    else:
        folder = folder.resolve()

    if not folder.is_dir():
        raise HTTPException(status_code=404, detail=f"Input folder does not exist: {folder}")

    try:
        # Load once before entering the image loop. Invalid models therefore fail
        # once instead of printing the same error for every image in the folder.
        lpr_service.ensure_models_loaded()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    started = time.perf_counter()
    files = iter_image_files(folder, recursive=request.recursive)
    namespace = uuid.uuid4().hex
    results = [
        lpr_service.recognize_image_path(
            image_path,
            preview=request.preview,
            preview_namespace=namespace,
        )
        for image_path in files
    ]

    return FolderRecognitionResponse(
        folder_path=str(folder),
        image_count=len(results),
        plate_count=sum(
            1 for result in results for plate in result.plates if plate.plate
        ),
        results=results,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 3),
    )


import shutil
import tempfile
import time
import uuid
from pathlib import Path

import cv2
from fastapi import File, HTTPException, UploadFile


@app.post("/api/v1/lpr/video")
def recognize_video(
    video: UploadFile = File(...),
    preview :bool =True,
    
) -> dict:
    filename = video.filename or "uploaded_video.mp4"
    suffix = Path(filename).suffix.lower()

    if suffix != ".mp4":
        raise HTTPException(
            status_code=400,
            detail="Only MP4 video files are supported.",
        )

    try:
        lpr_service.ensure_models_loaded()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc

    started = time.perf_counter()
    namespace = uuid.uuid4().hex

    results: list[dict] = []
    unique_plates: set[str] = set()

    frame_index = 0
    processed_frame_count = 0

    try:
        with tempfile.TemporaryDirectory(
            prefix="iranian_lpr_video_"
        ) as temporary_directory:
            temporary_directory_path = Path(temporary_directory)

            uploaded_video_path = (
                temporary_directory_path / filename
            )

            # Save the uploaded MP4 temporarily.
            with uploaded_video_path.open("wb") as destination:
                shutil.copyfileobj(video.file, destination)

            capture = cv2.VideoCapture(
                str(uploaded_video_path)
            )

            if not capture.isOpened():
                raise HTTPException(
                    status_code=400,
                    detail="Could not open the uploaded MP4 video.",
                )

            fps = float(
                capture.get(cv2.CAP_PROP_FPS) or 0
            )

            total_frames = int(
                capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            )

            try:
                while True:
                    success, frame = capture.read()

                    if not success:
                        break

                    current_frame_index = frame_index
                    frame_index += 1

                    frame_path = (
                        temporary_directory_path
                        / f"frame_{current_frame_index:08d}.jpg"
                    )

                    saved = cv2.imwrite(
                        str(frame_path),
                        frame,
                    )


                    if not saved:
                        print(
                            "[LPR][VIDEO][ERROR] "
                            f"Could not save frame "
                            f"{current_frame_index}"
                        )
                        continue

                    try:
                        recognition = (
                            lpr_service.recognize_image_path(
                                frame_path,
                                preview=preview,
                                preview_namespace=namespace,
                            )
                        )

                        detected_plates = [
                            prediction.plate
                            for prediction in recognition.plates
                            if prediction.plate
                        ]

                        for label in detected_plates:
                            unique_plates.add(label)

                        timestamp_ms = (
                            round(
                                current_frame_index
                                / fps
                                * 1000,
                                3,
                            )
                            if fps > 0
                            else None
                        )

                        print(
                            "[LPR][VIDEO] "
                            f"frame={current_frame_index} "
                            f"time_ms={timestamp_ms} "
                            f"plates="
                            f"{detected_plates or 'no plate'}"
                        )

                        results.append(
                            {
                                "frame_index": (
                                    current_frame_index
                                ),
                                "timestamp_ms": timestamp_ms,
                                "plates": detected_plates,
                                "recognition": (
                                    recognition.model_dump()
                                    if hasattr(
                                        recognition,
                                        "model_dump",
                                    )
                                    else recognition
                                ),
                            }
                        )

                        processed_frame_count += 1

                    except Exception as exc:
                        print(
                            "[LPR][VIDEO][ERROR] "
                            f"frame={current_frame_index} "
                            f"-> {exc}"
                        )

                        results.append(
                            {
                                "frame_index": (
                                    current_frame_index
                                ),
                                "timestamp_ms": (
                                    round(
                                        current_frame_index
                                        / fps
                                        * 1000,
                                        3,
                                    )
                                    if fps > 0
                                    else None
                                ),
                                "plates": [],
                                "error": str(exc),
                            }
                        )

            finally:
                capture.release()

    finally:
        video.file.close()

    plate_count = sum(
        len(result.get("plates", []))
        for result in results
    )

    return {
        "filename": filename,
        "fps": fps,
        "total_frames": total_frames,
        "processed_frame_count": processed_frame_count,
        "plate_count": plate_count,
        "unique_plates": sorted(unique_plates),
        "results": results,
        "elapsed_ms": round(
            (time.perf_counter() - started) * 1000,
            3,
        ),
    }
@app.post(
    "/api/v1/lpr/watch/start",
    response_model=WatchJobResponse,
    status_code=status.HTTP_201_CREATED,
)
def start_folder_watch(request: WatchStartRequest) -> WatchJobResponse:
    folder = Path(request.folder_path).expanduser()
    if not folder.is_absolute():
        folder = (settings.default_input_dir.parent / folder).resolve()
    try:
        return watch_manager.start(
            folder_path=folder,
            preview=request.preview,
            recursive=request.recursive,
            process_existing=request.process_existing,
            poll_interval_ms=request.poll_interval_ms,
        )
    except NotADirectoryError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/v1/lpr/watch/{job_id}", response_model=WatchJobResponse)
def get_watch_job(job_id: str) -> WatchJobResponse:
    try:
        return watch_manager.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Watch job not found") from exc


@app.post("/api/v1/lpr/watch/{job_id}/stop", response_model=StopWatchResponse)
def stop_watch_job(job_id: str) -> StopWatchResponse:
    try:
        stopped = watch_manager.stop(job_id)
        return StopWatchResponse(job_id=job_id, stopped=stopped)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Watch job not found") from exc
