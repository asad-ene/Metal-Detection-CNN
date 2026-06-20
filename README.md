# Metal Surface Defect Detection

A CNN-based classifier for detecting surface defects in steel, trained on the NEU-DET dataset. Classifies images into 6 defect categories using a fine-tuned ResNet18, with a Streamlit web app for interactive detection.

## Dataset

[NEU Surface Defect Database](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html) — 6 classes: crazing, inclusion, patches, pitted surface, rolled-in scale, scratches.

## Model

- **Backbone:** ResNet18 pretrained on ImageNet
- **Input size:** 128×128
- **Output:** 6 classes
- **Loss:** CrossEntropyLoss
- **Optimizer:** Adam (lr=0.001)

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning Rate | 0.001 |
| Batch Size | 32 |
| Epochs | 20 |
| Input Size | 128×128 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Augmentation | RandomHorizontalFlip, RandomRotation(10) |

## Folder Structure

```
NEU-DET/
├── train/
│   └── images/
│       ├── crazing/
│       ├── inclusion/
│       └── ...
└── validation/
    └── images/
        ├── crazing/
        ├── inclusion/
        └── ...
```

## Usage

Saves the best checkpoint (by validation accuracy) to `defect_model.pth`, along with the class list used during training. Also writes `classes.json` as a human-readable reference.

**Predict a single image (script):**
```python
predict_image('path/to/image.jpg', model, dataset)
```

**Run the web app:**
```bash
pip install -r requirements.txt
streamlit run (file path).py
```
Place `defect_model.pth` in the same folder as `app.py` before launching, or the app will run in demo mode with placeholder predictions. If you add or retrain the checkpoint while the app is already running, restart the Streamlit server (not just the browser tab) to pick up the new weights.

The app supports single or batch image uploads, displays the predicted defect prominently after detection, and lets you download a CSV log of all detections from the session via the button in the top right.

## Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | ~90–93% |

## Notes

- GPU recommended, training on CPU is significantly slower
- If no GPU available, use Google Colab with the dataset mounted from Google Drive
- Predictions below the confidence threshold (default 0.85) are flagged as `UNCERTAIN` for human review, both in the script logging and in the app
