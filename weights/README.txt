PASTE YOUR LOCAL MODELS HERE
============================

1. Ultralytics plate detector:

   weights/plate_detector.pt

2. Complete locally saved Hezar CRNN recognizer directory:

   weights/plate_recognizer/
   ├── model.pt
   ├── model_config.yaml
   └── preprocessor/
       └── image_processor_config.yaml

Do not paste a standalone character checkpoint as:

   weights/plate_char_recognizer.pt

The recognizer is loaded with:

   Model.load("weights/plate_recognizer", load_locally=True)

This project does not download either model. The YAML files above are model metadata
that must come with your Hezar checkpoint; the application itself has no YAML config.

Custom paths:

PLATE_DETECTOR_WEIGHTS=C:/models/plate_detector.pt
PLATE_RECOGNIZER_DIR=C:/models/plate_recognizer

For a multi-class detector whose plate class cannot be inferred by name:

LPR_PLATE_CLASS_IDS=1
