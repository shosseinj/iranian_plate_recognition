# Iranian LPR FastAPI — Backend Only

A small FastAPI service for Iranian license-plate recognition from:

- One uploaded image
- Every image already present in a server-side folder
- New images arriving in a folder in real time

There is no application frontend and no model downloader. FastAPI's `/docs` page remains available only for API testing.

## Recognition pipeline

```text
image
  -> Ultralytics YOLO plate detector
  -> cropped plate
  -> local Hezar CRNN whole-plate OCR
  -> normalized plate number
```

The service prints results directly:

```text
[LPR] 230.png -> 12ب34567
[LPR] 231.png -> no plate
```

## Model placement

Paste the detector here:

```text
weights/plate_detector.pt
```

Paste the **complete locally saved Hezar model directory** here:

```text
weights/plate_recognizer/
├── model.pt
├── model_config.yaml
└── preprocessor/
    └── image_processor_config.yaml
```

Do not use this old layout:

```text
weights/plate_char_recognizer.pt
```

A standalone `.pt` file does not contain all the Hezar architecture, label, preprocessing, and CTC-decoding metadata. The project itself has no YAML application configuration, but the complete Hezar checkpoint directory must include the metadata files belonging to that model.

No model is downloaded by this project.

Custom locations can be supplied with environment variables:

```powershell
$env:PLATE_DETECTOR_WEIGHTS="C:\models\plate_detector.pt"
$env:PLATE_RECOGNIZER_DIR="C:\models\plate_recognizer"
```

The recognizer is explicitly loaded locally:

```python
Model.load(str(recognizer_dir), load_locally=True)
```

This prevents a Windows path from being interpreted as a Hugging Face repository ID.

## Install

Python 3.11 or newer:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Use CPU:

```powershell
$env:LPR_DEVICE="cpu"
```

Use the first CUDA GPU:

```powershell
$env:LPR_DEVICE="0"
```

## Run

```powershell
python run.py
```

Or:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Health check

```http
GET /health
```

The response reports:

- Whether `plate_detector.pt` exists
- Whether the Hezar directory exists
- Which required Hezar files are missing
- Whether both models loaded
- The first model-loading error, if one occurred

## Process a folder once

```http
POST /api/v1/lpr/folder
Content-Type: application/json
```

```json
{
  "folder_path": "input_images",
  "preview": false,
  "recursive": false
}
```

The models are loaded before entering the image loop. An invalid model therefore returns one HTTP 503 error instead of printing the same error for every image.

PowerShell:

```powershell
$body = @{
    folder_path = "input_images"
    preview = $false
    recursive = $false
} | ConvertTo-Json

Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:8000/api/v1/lpr/folder" `
    -ContentType "application/json" `
    -Body $body
```

## Real-time folder watcher

```http
POST /api/v1/lpr/watch/start
Content-Type: application/json
```

```json
{
  "folder_path": "input_images",
  "preview": false,
  "recursive": false,
  "process_existing": true,
  "poll_interval_ms": 250
}
```

The start request validates and loads both models before creating the background watcher.

Read status:

```http
GET /api/v1/lpr/watch/{job_id}
```

Stop:

```http
POST /api/v1/lpr/watch/{job_id}/stop
```

## Upload one image

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/v1/lpr/image?preview=false" `
  -F "file=@sample.jpg"
```

## Preview flag

With `preview: false`:

- Detection and OCR run
- Plate numbers are printed and returned
- No annotated images are written

With `preview: true`:

- Annotated images are saved under `output/previews/<request-id>/`
- The response includes `preview_path` and `preview_url`

To also open a local OpenCV window:

```powershell
$env:LPR_SHOW_WINDOW="true"
```

Keep it false on Docker and headless servers.

## Direct Python integration

```python
from pathlib import Path

from app.services.lpr import LPRService

lpr = LPRService()
lpr.ensure_models_loaded()

result = lpr.recognize_image_path(
    Path("input_images/230.png"),
    preview=False,
)

for detected in result.plates:
    print(detected.plate)
```

## Run without HTTP

Process a folder:

```powershell
python scripts/process_folder.py --folder input_images
```

Process and save preview files:

```powershell
python scripts/process_folder.py --folder input_images --preview
```

Watch new images:

```powershell
python scripts/watch_folder.py --folder input_images
```

## Main environment variables

| Variable | Default | Purpose |
|---|---|---|
| `PLATE_DETECTOR_WEIGHTS` | `weights/plate_detector.pt` | Ultralytics plate detector |
| `PLATE_RECOGNIZER_DIR` | `weights/plate_recognizer` | Complete local Hezar directory |
| `LPR_DEVICE` | `0` | `cpu`, GPU index, or CUDA device |
| `LPR_USE_FP16` | `true` | YOLO FP16 inference on CUDA |
| `LPR_DETECTOR_IMGSZ` | `640` | Detector input size |
| `LPR_DETECTOR_CONFIDENCE` | `0.35` | Detector threshold |
| `LPR_DETECTOR_IOU` | `0.45` | Detector NMS IoU |
| `LPR_PLATE_CLASS_IDS` | empty | Optional explicit plate class IDs |
| `LPR_SHOW_WINDOW` | `false` | Open local OpenCV windows |
| `LPR_OUTPUT_PERSIAN_DIGITS` | `false` | Return Persian digit glyphs |

## Tests

```powershell
pip install -r requirements-dev.txt
pytest -q
```

The tests use fake models and do not require checkpoints.
