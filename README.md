# Metal Surface Defect Detection

A CNN-based classifier for detecting surface defects in steel, trained on the NEU-DET dataset. Classifies images into 6 defect categories using a fine-tuned ResNet18.

## Dataset

[NEU Surface Defect Database](http://faculty.neu.edu.cn/yunhyan/NEU_surface_defect_database.html) — 6 classes: crazing, inclusion, patches, pitted surface, rolled-in scale, scratches.

## Model

- **Backbone:** ResNet18 pretrained on ImageNet
- **Input size:** 224×224
- **Output:** 6 classes
- **Loss:** CrossEntropyLoss
- **Optimizer:** Adam (lr=0.001)

## Requirements

```bash
pip install torch torchvision scikit-learn Pillow
```

## Folder Structure

```
NEU-DET/
├── train/
│   └── images/
│       ├── crazing/
│       ├── inclusion/
│       └── ...
└── valid/
    └── images/
        ├── crazing/
        ├── inclusion/
        └── ...
```

## Usage

**Train:**
```bash
python train.py
```

**Predict a single image:**
```python
predict_image('path/to/image.jpg', model, dataset)
```

## Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | ~90%+ |

## Notes

- GPU recommended — training on CPU is significantly slower
- If no GPU available, use Google Colab with the dataset mounted from Google Drive
