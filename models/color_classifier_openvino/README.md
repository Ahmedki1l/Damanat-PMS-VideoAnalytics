# Color classifier (WS-B) — OpenVINO artifacts

* Backbone: MobileNetV3-Small
* Classes (11): black, white, grey, silver, red, blue, green, yellow, brown, beige, other
* Input: 224x224 RGB, ImageNet-normalised, letterbox-resized (BGR -> RGB inside the plugin)
* Trained for 20 epochs (early-stopped on val acc)
* Best val accuracy: 0.8344
* Held-out test accuracy: 0.7725
* Dataset rows — train: 644, val: 127, test: 153

## Files
* `model.xml` / `model.bin` — OpenVINO IR (consumed by `src.classifiers.color_classifier.OpenVINOColorClassifier`).
* `model.onnx` — intermediate ONNX export (kept for reproducibility).
* `best.pt` — torch state dict at the best-val-acc epoch.
* `labels.json` — `{labels, input_size, preprocessing}` consumed by the plugin at load time.
