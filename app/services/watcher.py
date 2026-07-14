from __future__ import annotations

import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from app.config import Settings, settings
from app.schemas import ImageRecognitionResult, WatchJobResponse
from app.services.lpr import LPRService, iter_image_files


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(slots=True)
class WatchJob:
    job_id: str
    folder_path: Path
    preview: bool
    recursive: bool
    process_existing: bool
    poll_interval_ms: int
    max_results: int
    status: str = "running"
    processed_images: int = 0
    detected_plates: int = 0
    started_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    error: str | None = None
    stop_event: threading.Event = field(default_factory=threading.Event)
    recent_results: deque[ImageRecognitionResult] = field(init=False)
    seen: set[Path] = field(default_factory=set)
    thread: threading.Thread | None = None

    def __post_init__(self) -> None:
        self.recent_results = deque(maxlen=self.max_results)

    def to_response(self) -> WatchJobResponse:
        return WatchJobResponse(
            job_id=self.job_id,
            status=self.status,  # type: ignore[arg-type]
            folder_path=str(self.folder_path),
            preview=self.preview,
            processed_images=self.processed_images,
            detected_plates=self.detected_plates,
            started_at=self.started_at,
            updated_at=self.updated_at,
            error=self.error,
            recent_results=list(self.recent_results),
        )


class FolderWatchManager:
    def __init__(
        self,
        lpr_service: LPRService,
        app_settings: Settings = settings,
    ) -> None:
        self.lpr_service = lpr_service
        self.settings = app_settings
        self._jobs: dict[str, WatchJob] = {}
        self._lock = threading.Lock()

    def start(
        self,
        folder_path: Path,
        preview: bool,
        recursive: bool,
        process_existing: bool,
        poll_interval_ms: int,
    ) -> WatchJobResponse:
        folder_path = folder_path.expanduser().resolve()
        if not folder_path.is_dir():
            raise NotADirectoryError(f"Input folder does not exist: {folder_path}")

        # Validate and load both models before starting the background thread.
        # The API call fails immediately if the local Hezar directory is invalid.
        self.lpr_service.ensure_models_loaded()

        job = WatchJob(
            job_id=uuid.uuid4().hex,
            folder_path=folder_path,
            preview=preview,
            recursive=recursive,
            process_existing=process_existing,
            poll_interval_ms=poll_interval_ms,
            max_results=self.settings.max_recent_watch_results,
        )
        if not process_existing:
            job.seen.update(iter_image_files(folder_path, recursive=recursive))

        job.thread = threading.Thread(
            target=self._run,
            args=(job,),
            name=f"lpr-watch-{job.job_id[:8]}",
            daemon=True,
        )
        with self._lock:
            self._jobs[job.job_id] = job
        job.thread.start()
        return job.to_response()

    def get(self, job_id: str) -> WatchJobResponse:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job.to_response()

    def stop(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        job.stop_event.set()
        if job.thread and job.thread.is_alive():
            job.thread.join(timeout=3.0)
        if job.status == "running":
            job.status = "stopped"
            job.updated_at = utc_now()
        return True

    def _run(self, job: WatchJob) -> None:
        print(f"[LPR][WATCH] started {job.job_id} for {job.folder_path}", flush=True)
        try:
            while not job.stop_event.is_set():
                files = iter_image_files(job.folder_path, recursive=job.recursive)
                new_files = [path for path in files if path not in job.seen]
                for image_path in new_files:
                    if job.stop_event.is_set():
                        break
                    if not self._wait_until_stable(image_path, job.stop_event):
                        continue
                    result = self.lpr_service.recognize_image_path(
                        image_path,
                        preview=job.preview,
                        preview_namespace=job.job_id,
                    )
                    job.seen.add(image_path)
                    job.processed_images += 1
                    job.detected_plates += len(
                        [plate for plate in result.plates if plate.plate]
                    )
                    job.recent_results.append(result)
                    job.updated_at = utc_now()

                job.stop_event.wait(job.poll_interval_ms / 1000.0)

            job.status = "stopped"
            job.updated_at = utc_now()
            print(f"[LPR][WATCH] stopped {job.job_id}", flush=True)
        except Exception as exc:
            job.status = "failed"
            job.error = str(exc)
            job.updated_at = utc_now()
            print(f"[LPR][WATCH][ERROR] {job.job_id}: {exc}", flush=True)

    @staticmethod
    def _wait_until_stable(path: Path, stop_event: threading.Event) -> bool:
        """Avoid reading a file while another process is still writing it."""
        previous_size = -1
        stable_checks = 0
        for _ in range(20):
            if stop_event.is_set() or not path.exists():
                return False
            current_size = path.stat().st_size
            if current_size > 0 and current_size == previous_size:
                stable_checks += 1
                if stable_checks >= 2:
                    return True
            else:
                stable_checks = 0
            previous_size = current_size
            time.sleep(0.05)
        return path.exists() and path.stat().st_size > 0
