# Iranian License Plate Recognition Service

This repository implements a FastAPI-based license-plate recognition pipeline for Iranian plates. It supports single-image inference, batch processing of images in a directory, and continuous processing of newly arriving images.

## Recognition Pipeline

```text
image
  -> YOLO plate detector
  -> plate crop
  -> local Hezar CRNN OCR
  -> normalized Iranian plate string
```

Both detector and recognizer are loaded from local model files, allowing the service to operate without downloading model weights at runtime.

## Features

- single-image upload endpoint;
- one-shot folder processing;
- background folder watcher for newly arriving images;
- CPU or CUDA inference;
- optional annotated preview images;
- explicit model-health reporting;
- reusable Python service class for direct integration.

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python run.py
```

Expected model layout:

```text
weights/
  plate_detector.pt
  plate_recognizer/
    model.pt
    model_config.yaml
    preprocessor/
      image_processor_config.yaml
```

## API

FastAPI automatically exposes interactive API documentation at `/docs`. A `/health` endpoint reports model availability, load status, and configuration problems before batch processing begins.

## Scope

This project focuses on building a modular recognition service around local deep-learning models. Model quality depends on the supplied detector/OCR checkpoints and should be evaluated separately on the target deployment data.


## Goal

The service separates model loading, plate detection, OCR, normalization, and folder monitoring behind a small API that can be tested without embedding the recognition pipeline in a larger application.

## Working with the Repository

Copy `.env.example` to a local `.env`, point it to the detector and complete Hezar recognizer directory, and check `/health` before submitting images. The API implementation is under `app/`, command-line helpers are under `scripts/`, and model layout guidance is in `weights/README.txt`.

Run the development tests with:

```bash
python -m pip install -r requirements-dev.txt
pytest
```

Use the upload endpoint for individual images and the folder endpoints only with server-side paths that the service is permitted to read. Generated previews and recognition outputs should be stored outside the source tree.
