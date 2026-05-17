# Type Classifier (OpenVINO IR)

MobileNetV3-Small body-type classifier — 6 classes.

## Classes (in IR output order)

  0. sedan
  1. suv
  2. hatchback
  3. pickup
  4. van
  5. bus

## Data dependency

This artifact was trained on real labelled crops produced by `tools/prepare_type_dataset.py` (data track D-3).

## Training metrics

- best_val_acc: 0.8973
- test_acc: 0.9070
- train_seconds: 229.7652

## Files

- `model.xml` / `model.bin` — OpenVINO IR (FP32).
- `model.onnx` — intermediate ONNX export.
- `labels.json` — class index → label map plus preprocessing constants.
- `README.md` — this file.

## Loading

```python
from src.classifiers.type_classifier import OpenVINOTypeClassifier
clf = OpenVINOTypeClassifier('models/type_classifier_openvino/model.xml')
label, conf = clf.predict(crop_bgr)
```

Retrain via `python tools/train_type_classifier.py --help`.
