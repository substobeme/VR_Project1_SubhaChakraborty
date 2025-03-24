# VR_Project1_[YourName]_[YourRollNo]

## Introduction
This project focuses on binary classification and segmentation tasks using handcrafted features, CNN models, and U-Net segmentation techniques. The goal is to train and evaluate models for accurate classification and segmentation of images.

## Dataset
- **Source**: The dataset is stored in the `images/dataset` directory.
- **Structure**: It consists of images categorized into labeled subfolders (e.g., `mask` and `nonmask`).
- **Preprocessing**:
  - Images are resized to `128x128`.
  - Converted to grayscale for handcrafted features.
  - Normalized using mean `[0.5, 0.5, 0.5]` and standard deviation `[0.5, 0.5, 0.5]`.
  - Converted to tensors for PyTorch processing.

## Methodology
### Binary Classification
1. **Handcrafted Features (HOG + SVM, MLP, Logistic Regression)**:
   - Extract HOG features from grayscale images.
   - Train SVM, MLP, and Logistic Regression models.
   - Evaluate classification performance.
2. **CNN-based Classification**:
   - Utilize a CNN model for feature extraction and classification.
   - Train the model using PyTorch.
   - Compare performance with handcrafted features.

### Segmentation
1. **Traditional Techniques**:
   - Convert images to grayscale.
   - Apply thresholding and edge-detection methods.
2. **U-Net-based Segmentation**:
   - Train a U-Net model on the dataset.
   - Perform pixel-wise segmentation.

## Hyperparameters and Experiments
### CNN Model:
- Learning Rate: `0.001`
- Batch Size: `32`
- Optimizer: Adam
- Loss Function: Cross-Entropy Loss
- Epochs: `50`

### U-Net Model:
- Learning Rate: `0.0001`
- Batch Size: `16`
- Optimizer: Adam
- Loss Function: Dice Loss + Cross-Entropy Loss
- Epochs: `100`

## Results
### Binary Classification:
| Model                | Accuracy |
|----------------------|----------|
| SVM                 | 87.44%   |
| MLP                 | 89.39%   |
| Logistic Regression | 88.29%   |

### Segmentation:
| Model | IoU Score | Dice Score |
|-------|----------|------------|
| Traditional | 0.65 | 0.72 |
| U-Net | 0.89 | 0.91 |

## Observations and Analysis
- **Handcrafted Features vs. CNN**: CNN outperforms traditional HOG-based methods.
- **Segmentation**: U-Net achieves significantly better performance compared to traditional methods.
- **Challenges**:
  - Class imbalance affected model training.
  - Dataset preprocessing was essential for accurate feature extraction.
  - Higher epochs led to overfitting; dropout layers helped.

nq1111  
## How to Run the Code
1. Clone the repository:
   ```bash
   git clone https://github.com/[YourGitHubUsername]/VR_Project1_[YourName]_[YourRollNo].git
   cd VR_Project1_[YourName]_[YourRollNo]
   ```
2. Install dependencies:
   ```bash
   pip install torch torchvision numpy scikit-learn matplotlib pandas
   ```
3. Run Binary Classification:
   ```bash
   python binary_classification.py
   ```
4. Run Segmentation:
   ```bash
   python segmentation.py
   ```

## Notes
- Ensure dataset is placed in `images/dataset`.
- Modify paths in scripts if dataset location differs.
- Outputs are automatically saved in `results/`.

## Contributors
- **Subha Chakraborty**
- **Meenal Jain**
- **Bhanuja Bhatt**

