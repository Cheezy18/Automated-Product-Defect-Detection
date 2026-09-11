# Automated Product Defect Detection

## Objective
Build a Computer Vision system to automatically detect product defects using binary image classification.

## Dataset
The project uses a subset of the AITEX fabric image dataset containing **120 images**:
- 60 defective images
- 60 normal images
- 96 training images
- 24 validation images

The dataset is not included in this repository.

## Methodology

1. Prepare the dataset.
2. Resize images to **224 × 224**.
3. Convert grayscale images to 3 channels.
4. Convert images to tensors.
5. Normalize the images.
6. No random augmentation is used.
7. Use a binary image classification model.
8. Use a pretrained **ResNet18** model.
9. Freeze the pretrained layers.
10. Replace the final fully connected layer with a 2-class classifier.
11. Train the model.
12. Evaluate using Accuracy, Precision, Recall and F1-score.
13. Generate a confusion matrix.
14. Perform error analysis using misclassified images.

## Model

**Pretrained ResNet18**

The pretrained ResNet18 feature extraction layers are frozen. Only the final fully connected classification layer is trainable.

Classes:
- Defective
- Normal

## Training

- Loss function: CrossEntropyLoss
- Optimizer: Adam
- Learning rate: 0.001
- Batch size: 16
- Epochs: 10

## Results

| Metric | Score |
|---|---:|
| Training Accuracy | 67.71% |
| Validation Accuracy | 70.83% |
| Precision | 64.71% |
| Recall | 91.67% |
| F1-Score | 75.86% |

The validation confusion matrix contained:
- 11 correctly classified defective images
- 1 defective image classified as normal
- 6 normal images classified as defective
- 6 correctly classified normal images

## Error Analysis

Misclassified images were examined to identify possible reasons for incorrect predictions. Possible causes include:
- subtle defects
- similar visual appearance between normal and defective samples
- lighting or texture variations
- background/fabric texture
- limited training examples

## Repository Structure

```text
Automated-Product-Defect-Detection/
│
├── README.md
├── main.py
├── requirements.txt
│
├── model/
│   └── train-model.pth
│
└── results/
    ├── sample_training_images.png
    ├── training_validation_loss.png
    ├── training_validation_accuracy.png
    ├── confusion_matrix.png
    ├── misclassified_images.png
    ├── error_analysis.csv
    └── metrics.csv
```

## How to Run

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Extract the dataset using the structure described above.

3. Set the dataset path if necessary:

```bash
set DATA_DIR=path/to/aitex_120_fast/aitex_120_fast
```

On Linux/macOS:

```bash
export DATA_DIR=path/to/aitex_120_fast/aitex_120_fast
```

4. Run:

```bash
python main.py
```

The trained model will be saved inside the `model/` folder and result images/tables will be saved inside `results/`.

