# Workout Video Classification with R3D-18

This project focuses on classifying workout videos into exercise categories using a 3D convolutional neural network (R3D-18) trained on short video clips.

## Dataset Overview

- **Input**: 456 workout videos across multiple categories.
- **Preprocessing**:
  - Each video is split into 6 clips of 16 consecutive frames.
  - Frames are resized to `128×171`, then center-cropped to `112×112`.
  - All frames are normalized using Kinetics-400 statistics.

## Model Architecture

- **Base model**: `r3d_18` pretrained on Kinetics-400.
- **Modifications**:
  - Replaced classification head with a single FC layer.
  - Unfroze the final residual block (`layer4`) for fine-tuning.
  - Applied differential learning rates: higher for FC head, lower for backbone.

## 🏋️ Training Setup

- **Epochs**: 10
- **batch_size**: 8
- **Optimizer**: Adam  
- **Learning rates**:
  - FC layer: `1e-3`
  - Unfrozen backbone: `1e-4`
- **Scheduler**: `StepLR` with step size 5 and gamma 0.1  
- **Loss**: CrossEntropyLoss with class weights  
- **Batch size**: 8  
- **Augmentations**:
  - Applied during both training and validation
  - Resize → CenterCrop → Normalize
  - *Note*: Temporal jittering and random flip were tested but removed due to no gain

## Results

- **Best validation accuracy**: ~90%
- - **Best Test accuracy**: 93.8%
- **Evaluation performed during validation**:
  - Accuracy and loss computed at both **clip** and **video** level
  - Video-level predictions aggregated via softmax averaging

## How to Run

1. Upload the dataset to the desired directory
2. Open the notebook 
3. Run all cells to reproduce results (note: GPU runtime recommended)
4. Adjust paths in the notebook if needed

## Notes

- Some augmentations and architectures were tested (e.g., MLP head, temporal jittering), but excluded after performance analysis.
- Seed sensitivity and cross-validation are planned for future work.
