# Ovarian Cancer Subtype Classification

> **Description:** A deep learning pipeline that classifies ovarian histopathology images into four cancer subtypes (Clear Cell, Endometrioid, Mucinous, Serous) or non-cancerous tissue, using ResNet50V2 transfer learning with Optuna hyperparameter tuning.

## Overview

Ovarian cancer is the fifth deadliest cancer in women, and early, accurate histopathological classification matters for treatment decisions. This project trains a convolutional neural network on histopathology images to tell five categories apart:

| Class label | Meaning |
|---|---|
| `Clear_Cell` | Clear cell carcinoma |
| `Endometri` | Endometrioid carcinoma |
| `Mucinous` | Mucinous carcinoma |
| `Serous` | Serous carcinoma |
| `Non_Cancerous` | Non-cancerous tissue |

## Results

| Metric | Value |
|---|---|
| Best test accuracy (5-class) | **94.57%** |
| Additional run (tiling-based experiment) | ~78% |

**Per-class accuracy on the best run:** Mucinous 98%, Non_Cancerous 98%, Endometri 93%, Clear_Cell 95%, Serous 87%.

**Main error patterns:**
- Clear_Cell and Mucinous were confused in both directions.
- Serous was occasionally mistaken for Clear_Cell and Mucinous.
- Endometri showed minor confusion with Clear_Cell and Serous.
- In the lower-scoring run, Mucinous vs. Non_Cancerous was the most common confusion, and class imbalance (especially Endometri) hurt precision.

## Dataset

[Ovarian Cancer and Subtypes Dataset – Histopathology](https://www.kaggle.com/datasets/bitsnpieces/ovarian-cancer-and-subtypes-dataset-histopathology) from Kaggle (`bitsnpieces`). Images are organized in one folder per class, which Keras reads directly.

## Method

1. **Data loading:** `ImageDataGenerator.flow_from_directory`, images resized to 224×224, batch size 16, 80/20 train/validation split.
2. **Augmentation:** rotation (15°), width/height shift (0.1), shear (0.1), zoom (0.1), horizontal flip, brightness range (0.9–1.1).
3. **Class imbalance:** balanced class weights computed with scikit-learn and passed to `model.fit`.
4. **Model:** ImageNet-pretrained **ResNet50V2** (last 10 layers unfrozen) → Global Average Pooling → Dense(512, ReLU, L2) → BatchNorm → Dropout → Dense(5, softmax).
5. **Optimization:** SGD with momentum 0.9. **Optuna** runs 10 trials, searching dropout (0.3–0.5) and learning rate (1e-5 to 1e-3, log scale), maximizing validation accuracy.
6. **Training controls:** up to 30 epochs per trial, `EarlyStopping` (patience 15), `ReduceLROnPlateau`, and `ModelCheckpoint` saving the best weights per trial.
7. **Evaluation:** best trial is reloaded, test accuracy computed, confusion matrix plotted, and a simple train-vs-validation overfitting check printed.

## Getting Started

The script was exported from Google Colab and is easiest to run there.

```bash
pip install tensorflow scikit-learn matplotlib numpy optuna kagglehub kaggle
```

1. Get a Kaggle API token (`kaggle.json`) and place it in the working directory.
2. Run the script. It downloads and unzips the dataset to `/content/OvarianCancer/`.
3. Adjust `train_dir` / `test_dir` if your data lives elsewhere.

Colab shell commands (lines starting with `!`) need to be removed or replaced if you run it as a plain `.py` file.

## Known Limitations

- **Test set overlaps training data.** `test_dir` and `train_dir` point to the same folder, so the "test" images include the ones used for training and validation. The reported test accuracy is likely optimistic. Splitting into separate train/validation/test folders (ideally by patient or slide) is the most important next fix.
- **Class imbalance:** some subtypes, notably Endometri, have fewer samples.
- **Image-level, not slide-level:** the model classifies individual images, and clinical use would require adapting to whole-slide images (WSI).
- Research prototype only; not validated for clinical use.

## Future Work

- Augment or resample underrepresented classes and refine tiling strategies.
- Improve feature extraction for Clear_Cell, Mucinous, and Serous.
- Explore other transfer-learning backbones and hybrid models.
- Validate on external, real-world clinical datasets.

## Tech Stack

Python, TensorFlow/Keras, ResNet50V2, Optuna, scikit-learn, Matplotlib, KaggleHub
