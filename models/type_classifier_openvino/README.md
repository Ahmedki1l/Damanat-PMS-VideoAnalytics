# Type Classifier (OpenVINO IR) — placeholder

This directory will hold the OpenVINO IR for the 6-class body-type
classifier (sedan, SUV, hatchback, pickup, van, bus).

The artifact is produced by:

```
python tools/train_type_classifier.py
```

That command writes:

- `model.xml` / `model.bin` — OpenVINO IR (FP32).
- `model.onnx` — intermediate ONNX export.
- `labels.json` — class index → label map plus preprocessing constants.
- `README.md` — auto-generated, replaces this placeholder.
- `best.pt` — torch checkpoint used to export the IR.

## Data dependency

`tools/train_type_classifier.py` consumes confirmed labels from
`tests/data/type_classifier/{train,val,test}.csv` produced by
`tools/prepare_type_dataset.py`. When real labels are sparse, the trainer
falls back to a synthetic geometric-primitives dataset so Phase 2 still has
a working IR to integrate against. The fallback model is intentionally
degraded — replace it once data-track D-3 lands.

## Loading from Python

```python
from src.classifiers.type_classifier import OpenVINOTypeClassifier

clf = OpenVINOTypeClassifier()  # defaults to models/type_classifier_openvino/model.xml
label, conf = clf.predict(crop_bgr)
```
