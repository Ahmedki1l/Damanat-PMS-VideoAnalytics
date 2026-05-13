# Color classifier (WS-B) — OpenVINO artifacts

* Backbone: MobileNetV3-Small
* Classes (11): black, white, grey, silver, red, blue, green, yellow, brown, beige, other
* Input: 96x96 RGB, ImageNet-normalised, letterbox-resized (BGR -> RGB inside the plugin)
* Trained for 12 epochs (early-stopped on val acc)
* Best val accuracy: 0.9375
* Held-out test accuracy: 0.9688
* Dataset rows — train: 616, val: 132, test: 132

## Files
* `model.xml` / `model.bin` — OpenVINO IR (consumed by `src.classifiers.color_classifier.OpenVINOColorClassifier`).
* `model.onnx` — intermediate ONNX export (kept for reproducibility).
* `best.pt` — torch state dict at the best-val-acc epoch.
* `labels.json` — `{labels, input_size, preprocessing}` consumed by the plugin at load time.


## Fallback synthetic data

This artifact was trained on the **synthetic swatch fallback** (see `tools/train_color_classifier.py::build_synthetic_manifest`)
because no human-labelled manifest was available at training time.
Accuracy is intentionally low and the model is suitable only for pipeline smoke-testing, not production matching. Re-train on the real D-2 deliverable before relying on this output.
