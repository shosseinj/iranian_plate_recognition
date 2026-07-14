from __future__ import annotations

import argparse
import time
from pathlib import Path

from app.services.lpr import LPRService
from app.services.watcher import FolderWatchManager


def main() -> int:
    parser = argparse.ArgumentParser(description="Watch a folder and recognize new plate images.")
    parser.add_argument("--folder", default="input_images")
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--poll-ms", type=int, default=250)
    args = parser.parse_args()

    folder = Path(args.folder).expanduser().resolve()
    manager = FolderWatchManager(LPRService())
    job = manager.start(
        folder_path=folder,
        preview=args.preview,
        recursive=args.recursive,
        process_existing=not args.skip_existing,
        poll_interval_ms=args.poll_ms,
    )
    print(f"Watch job started: {job.job_id}. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        manager.stop(job.job_id)
        print("Watch stopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
