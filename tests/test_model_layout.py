from dataclasses import replace
from pathlib import Path

import pytest

from app.config import settings
from app.services.lpr import LPRService


def test_recognizer_requires_complete_local_directory(tmp_path: Path) -> None:
    detector = tmp_path / "plate_detector.pt"
    detector.write_bytes(b"placeholder")
    recognizer = tmp_path / "plate_recognizer"
    recognizer.mkdir()

    service = LPRService(
        replace(
            settings,
            detector_weights=detector,
            recognizer_model_dir=recognizer,
            output_dir=tmp_path / "output",
            device="cpu",
        )
    )

    missing = {path.relative_to(recognizer).as_posix() for path in service.recognizer_missing_files()}
    assert missing == {
        "model.pt",
        "model_config.yaml",
        "preprocessor/image_processor_config.yaml",
    }

    with pytest.raises(RuntimeError, match="local Hezar recognizer directory is incomplete"):
        service.ensure_models_loaded()

    # The original exception is cached, so a folder loop will not try loading the
    # same invalid model again for every image.
    with pytest.raises(RuntimeError, match="Model loading previously failed"):
        service.ensure_models_loaded()
