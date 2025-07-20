# Pet Adoption Prediction

## Overview

This project is a multimodal machine learning system designed to predict how quickly a pet will be adopted. It brings together different types of data - images, text descriptions, and structured information - to classify pets into categories based on their expected adoption speed.

## Table of Contents

1. [Overview](#overview)
2. [Problem Statement](#problem-statement)
3. [Dataset](#dataset)
4. [Project Structure](#project-structure)
5. [Model Categories](#model-categories)
6. [Installation](#installation)
7. [Usage](#usage)
8. [Model Performance](#model-performance)
9. [Output Structure](#output-structure)
11. [Key Innovations](#key-innovations)
12. [Contributing](#contributing)
13. [License](#license)

## Problem Statement

The goal is to predict how quickly a pet will be adopted based on various features:
- **Structured Data**: Age, breed, color, size, health status, vaccination status, and other categorical/numerical features
- **Text**: Pet descriptions and names
- **Images**: Pet photos

> [!IMPORTANT]
> Throughout this project, we use the following abbreviations:
> - **SD** = Structured Data 
> - **TD** = Text Data 
> - **ID** = Image Data

The target variable is Adoption Speed which is mapped to 4 classes:
- **Class 0**: Fast adoption (0-7 days)
- **Class 1**: Medium adoption (8-30 days) 
- **Class 2**: Slow adoption (31-90 days)
- **Class 3**: Very slow adoption (+100 days) 

## Dataset

The dataset comes from the Kaggle competition [PetFinder Adoption Prediction](https://www.kaggle.com/competitions/petfinder-adoption-prediction) and includes structured data, text descriptions, and pet images.  

We combine all these modalities to predict and classify the adoption speed of each pet.

> [!NOTE]
> - Since the image files cannot be uploaded to GitHub, download them directly from Kaggle and place them in: `original_datasets/images`
> - We performed preprocessing on the images and generated `.pkl` and `.npy` files required for running the models. To recreate these files, run: `original_datasets/convert_images_to_pkl`
> - These processed files will be generated under the `data_processed` directory.
> - Currently, in `data_processed`, you will see three placeholder files: `image_data.pkl`, `image.npy`, and `labels.npy`. These are **empty dummy files**. Replace them with the newly generated files. *(You can also refer to `data_processed_FREAD_FOR_IMAGES.txt` for more details.)*

## Project Structure

```
Pet_Adoption_Prediction/
├── dataset/                            # Data management and preprocessing
│   ├── data_processed/                 # Processed datasets ready for training
│   ├── original_datasets/              # Raw dataset files
│   └── splits/                         # Train/validation/test splits
├── models/                             # Model implementations organized by modality
│   ├── Baseline/                       # Baseline model
│   ├── SD/                             # SD models
│   ├── TD_AND_TD_with_SD/              # TD model; TD_and_DS models
│   ├── ID_AND_ID_with_SD/              # ID model; ID_and_DS models
│   ├── ID_with_TD/                     # ID_and_TD models
│   └── SD_with_TD_with_ID/             # SD_TD_ID models
├── README.md                           # This file
├── LICENSE                             # License information
└── requirements.txt                    # Python dependencies

# Each model subdirectory typically contains:
# ├── main.py                           # Main training script
# ├── model_trainer.py                  # Training logic
# ├── evaluator.py                      # Evaluation metrics
# └── result_saver.py                   # Save results and models
```

## Model Categories

| **Category**              | **Model**                    |
|---------------------------|------------------------------|
| **Baseline Model**        | Model 0                      |
| **Structured Data Models**| KNN                          |
|                           | SVM                          |
|                           | XGBoost                      |
|                           | AdaBoost                     |
| **Text Models**           | TF-IDF + Logistic Regression |
|                           | TF-IDF + SVM                 |
|                           | TF-IDF + XGBoost             |
|                           | TF-IDF + KNN                 |
|                           | BERT + MLP                   |
|                           | BERT + CNN                   |
|                           | BERT + BiLSTM                |
| **Image Models**          | CNN                          |
|                           | ResNet18                     |
|                           | VGG                          |
| **Bimodal Models**        | CNN + Logistic Regression    |
|                           | CNN + LSTM                   |
|                           | CNN + MLP                    |
| **Trimodal Models**       | CNN + MLP                    |
|                           | CNN + MLP + Attention        |
|                           | CNN + MLP + RoBERTa          |
|                           | CNN + Attention + MLP        |
|                           | CLIP + MLP                   |


## Installation

1. Clone the repository:
```bash
git clone https://github.com/OriGoldfrydCS/Pet_Adoption_Prediction.git
cd Pet_Adoption_Prediction

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running Individual Models

Each model category has its own main script. Examples:

```bash
# Baseline model
python models/baseline_model/model_0.py

# SD models
python models/structured_data_models/xgboost/main.py
python models/structured_data_models/svm/main.py

# TD models
python models/text_models_only_and_with_structured_data/logistic_text/main.py
python models/text_models_only_and_with_structured_data/mlp_text/main.py

# ID models
python models/image_models_only_and_with_structured_data/cnn_images/main.py
python models/image_models_only_and_with_structured_data/ResNet18_images/main.py

# Multimodal models
python models/structured_data_with_text_and_images/CNN_MLP/main.py
python models/structured_data_with_text_and_images/CNN_MLP_RoBERTa/main.py
```

## Model Performance

All models are evaluated with a comprehensive set of metrics when you run them individually, including:
- **Accuracy** – overall classification accuracy
- **Macro F1-Score** – F1 score averaged across all classes
- **Cross-Entropy Loss** – for neural network models
- **Precision and Recall** – per class, summarized in a detailed classification report
- **Confusion Matrix** – for visual analysis of class-level performance

Below is a comparison table showcasing only **Macro F1** and **Loss** values for a selected subset of models.  
*Note: This table is just a summary. When you run any model from this repository, you will receive the full set of metrics listed above.*

| **Rank** | **Data Type(s)**   | **Model**                           | **F1 (↑)** | **Loss (↓)** |
|---------:|-------------------|--------------------------------------|-----------:|-------------:|
| 1        | *SD ∪ TD*         | CNN (BERT Full)                      | **0.8207** | **0.4320**   |
| 2        | *SD ∪ TD*         | SVM (TF-IDF)                         | 0.7881     | 0.4931       |
| 3        | *SD ∪ TD ∪ ID*    | CLIP + MLP (Mean Img.)               | 0.6850     | 0.9375       |
| 4        | *SD ∪ TD ∪ ID*    | CLIP + MLP (10 Img.)                 | 0.6583     | 1.0581       |
| 5        | *TD*               | CNN (BERT Full)                      | 0.6567     | 0.8733       |
| 6        | *TD*               | SVM (BERT CLS)                       | 0.6483     | 0.7847       |
| 7        | *SD ∪ TD ∪ ID*    | CNN + MLP + Att (TF-IDF + All Img.)  | 0.6432     | 1.2133       |
| 8        | *SD ∪ TD ∪ ID*    | CNN + MLP + Att (TF-IDF + Mean Img.) | 0.6405     | 1.2818       |
| 9        | *SD ∪ ID*         | ResNet18 + MLP (Mean Img.)           | 0.6258     | 1.6322       |
| 10       | *SD ∪ ID*         | CNN + MLP (TF-IDF + Mean Img.)       | 0.6204     | 1.1364       |
| 11       | *TD*              | SVM (TF-IDF)                         | 0.5874     | 0.8511       |
| 12       | *SD*              | SVM                                  | 0.6177     | 0.9511       |
| 13       | *ID*              | ResNet18                             | 0.3920     | 4.3145       |
| 14       | Baseline          | -                                    | 0.1175     | 24.9660      |


> [!note]
> Results are automatically saved to timestamped directories under `performance_and_models/` for each model type.
> This directory is **not uploaded to GitHub** due to its large size and GitHub's file upload limitations.

## Output Structure

Each model run creates a timestamped directory containing:
- `model.pt` / `model.joblib`: Trained model weights
- `*_metrics.txt`: Performance metrics
- `*_classification_report.txt`: Detailed classification report
- `*_confusion_matrix.png`: Confusion matrix visualization
- `*_true.csv` / `*_pred.csv`: True and predicted labels
- `hyperparameters.txt`: Model architecture and parameters

## Key Innovations

1. **Comprehensive Multimodal Approach**: Systematic exploration of uni-, bi-, and tri-modal architectures
2. **Attention Mechanisms**: Both inter-modal (between modalities) and intra-modal (within image sequences)
3. **Multiple Text Representations**: TF-IDF, BERT, RoBERTa for diverse text understanding
4. **Image Sequence Handling**: Processing multiple images per pet with attention and averaging
5. **Systematic Evaluation**: Consistent metrics and output format across all models

## Contributing

This project follows a modular structure where each model type is self-contained with its own trainer, evaluator, and result saver components.

## License

See LICENSE file for details.