# Project: Face Mask Detection, Classification, and Segmentation


## Introduction
This project implements face mask detection using both traditional computer vision techniques and deep learning approaches. The project consists of two main tasks:

- Binary classification of face images as "mask" or "no mask" using handcrafted features and CNN
- Mask segmentation using traditional techniques and a U-Net architecture

## Dataset
The project uses two datasets:

- For binary classification: Dataset located in "https://github.com/chandrikadeb7/Face-Mask-Detection/tree/master/dataset" with two classes - "mask" and "nonmask"
- For segmentation: Dataset from "https://github.com/sadjadrz/MFSD"

## Methodology
### Task 1: Binary Classification
### Handcrafted Features Approach

#### Data Preparation:

- Loaded images from the dataset and converted them to grayscale
- Resized all images to 128×128 pixels
- Split the dataset into 80% training and 20% testing sets


#### Feature Extraction:

- Utilized Histogram of Oriented Gradients (HOG) to extract features
- HOG parameters: 9 orientations, 8×8 pixels per cell, 2×2 cells per block


#### Classification Models:

- Support Vector Machine (SVM) with linear kernel
- Multi-Layer Perceptron (MLP) with 128 and 64 neurons in hidden layers
- Logistic Regression
- Decision Tree
- Random Forest with 100 estimators



### CNN Approach

#### Architecture:

- Three convolutional blocks (each with Conv2D + ReLU + MaxPool)
- Input channels: 3 (RGB), Output channels: 32 → 64 → 128
- Fully connected layers: 1281616 → 128 → 1
- Sigmoid activation for binary classification


#### Training Details:

- Loss Function: Binary Cross Entropy
- Optimization: Adam optimizer with learning rate 0.001
- Batch size: 32
- Epochs: 10
- Data augmentation: Normalization with mean=[0.5, 0.5, 0.5] and std=[0.5, 0.5, 0.5]
- Learning rates: 0.0001, 0.001, 0.01
- Batch sizes: 32, 64, 128
- Optimizers: Adam, SGD, RMSprop



### Task 2: Mask Segmentation
### Traditional Segmentation Approach

#### Image Processing:

- Loaded original images and corresponding ground truth masks
- Converted images to grayscale
- Applied simple thresholding technique (binary threshold at value 127)


#### Intersection over Union (IoU):

Calculated Intersection over Union (IoU) between predicted masks and ground truth
Tracked number of images with IoU > 0.5



### U-Net Approach

#### Architecture:

- Encoder: 3 blocks with decreasing spatial dimensions and increasing channels (1→16→32→64)
- Bottleneck: Convolutional block with 128 channels
- Decoder: 3 blocks with skip connections from encoder
- Final 1×1 convolution with sigmoid activation


#### Training Details:

- Loss Function: Binary Cross Entropy
- Optimizer: Adam with learning rate 0.001 and weight decay 1e-5
- Batch size: 2 (optimized for CPU training)
- Image resolution: Resized to 32×32 for faster training
- Epochs: 5


#### Evaluation Metrics:

- Intersection over Union (IoU)
- Dice coefficient (F1 score)



## Hyperparameters and Experiments

### Binary Classification CNN
The CNN model was evaluated with various hyperparameter combinations. Below are the results categorized by learning rate.

#### Learning Rate: 0.0001

| Batch Size | Optimizer | Accuracy (%) |
|-----------|-----------|-------------|
| 32        | Adam      | 90.61       |
| 32        | SGD       | 54.39       |
| 32        | RMSprop   | 93.66       |
| 64        | Adam      | 93.17       |
| 64        | SGD       | 53.29       |
| 64        | RMSprop   | 91.71       |
| 128       | Adam      | 89.51       |
| 128       | SGD       | 53.29       |
| 128       | RMSprop   | 92.32       |

#### Learning Rate: 0.001

| Batch Size | Optimizer | Accuracy (%) |
|-----------|-----------|-------------|
| 32        | Adam      | 94.51       |
| 32        | SGD       | 88.66       |
| 32        | RMSprop   | 93.41       |
| 64        | Adam      | 94.51       |
| 64        | SGD       | 82.93       |
| 64        | RMSprop   | 88.17       |
| 128       | Adam      | 94.63       |
| 128       | SGD       | 75.12       |
| 128       | RMSprop   | 87.56       |

#### Learning Rate: 0.01

| Batch Size | Optimizer | Accuracy (%) |
|-----------|-----------|-------------|
| 32        | Adam      | 46.71       |
| 32        | SGD       | 90.24       |
| 32        | RMSprop   | 46.71       |
| 64        | Adam      | 53.29       |
| 64        | SGD       | 91.46       |
| 64        | RMSprop   | 46.71       |
| 128       | Adam      | 53.29       |
| 128       | SGD       | 89.63       |
| 128       | RMSprop   | 46.71       |

### U-Net Segmentation
The U-Net model was simplified for CPU training:

- Reduced channel dimensions (original: 64→128→256→512, simplified: 16→32→64→128)
- Smaller input resolution (32×32 instead of larger resolutions)
- Batch size of 2 to accommodate memory constraints

## Results

### Binary Classification

#### Handcrafted Features Results:

| Model               | Accuracy (%) | Precision (Class 0) | Recall (Class 0) | F1 Score (Class 0) | Precision (Class 1) | Recall (Class 1) | F1 Score (Class 1) |
|---------------------|-------------|----------------------|------------------|---------------------|----------------------|------------------|---------------------|
| SVM                | 87.44        | 0.91                 | 0.86             | 0.88                | 0.84                 | 0.89             | 0.86                |
| MLP                | 89.39        | 0.92                 | 0.89             | 0.90                | 0.87                 | 0.90             | 0.88                |
| Logistic Regression| 88.29        | 0.91                 | 0.88             | 0.89                | 0.85                 | 0.89             | 0.87                |
| Decision Tree      | 69.39        | 0.73                 | 0.72             | 0.72                | 0.65                 | 0.66             | 0.66                |
| Random Forest      | 88.41        | 0.87                 | 0.94             | 0.90                | 0.91                 | 0.82             | 0.86                |

#### CNN Results:

- Best performance: 97.32% accuracy with the initial CNN model configuration
- Best hyperparameter combination: LR=0.001, Batch Size=128, Optimizer=Adam (94.63%)

### Segmentation

#### Traditional Approach:

- Average IoU across all images: 0.2674
- Number of images with IoU > 0.5: 1,434 out of 9,382 images (15.28%)

#### U-Net Results:

- Average IoU: 0.8589
- Average Dice Score: 0.9218

## Observations and Analysis

### Binary Classification Insights:

- MLP performed the best among handcrafted feature approaches (89.39%)
- CNN significantly outperformed all handcrafted feature methods (97.32%)
- Decision Tree had the poorest performance (69.39%), suggesting the decision boundaries for this problem are not well-represented by axis-aligned splits
- Adam optimizer generally performed better than SGD and RMSprop for low learning rates
- Optimal learning rate was 0.001 across most configurations
- SGD performed better with higher learning rates (0.01) while Adam and RMSprop struggled


### Segmentation Insights:

- Simple thresholding performed poorly (0.2674 IoU), indicating that basic intensity-based segmentation is insufficient for mask detection
- U-Net dramatically improved segmentation quality (0.8589 IoU)
- Despite being simplified for CPU training, the U-Net architecture still achieved excellent results with just 5 epochs
- The high Dice score (0.9218) confirms that the U-Net model accurately captures the mask boundaries
  
## How to Run the Code
1. Clone the repository:
   ```bash
   git clone https://github.com/substobeme/VR_Assignment1_SubhaChakraborty_MT2024156.git
   cd VR_Assignment1_SubhaChakraborty_MT2024156
   ```
2. Install dependencies:
   ```bash
   pip install torch torchvision numpy scikit-learn matplotlib pandas
   ```
3. Run Binary Classification:
   ```bash
   python Q1&Q2.py
   ```
4. Run Segmentation:
   ```bash
   python 'Q3 & Q4.py'
   ```


## Contributors
- **Subha Chakraborty**
- **Meenal Jain**
- **Bhanuja Bhatt**

