# MicroProphet
MicroProphet is a transformer-based digital twin model for microbial time-series prediction, designed to accurately forecast microbial dynamics without requiring data alignment or imputation, making it suitable for diverse and irregular microbiome datasets.

![MicroProphet](MicroProphet.png)

# Installation

Install MicroProphet using pip:

```bash
cd microprophet
pip install .
```

Or you can download the offline installation package from the GitHub release page and install MicroProphet using the following command:
```bash
pip install micro_prophet-1.1.0-py3-none-any.whl
```

MicroProphet is developed under the environment of Python3, and uses Pytorch to build the model. GPU devices are recommended to accelerate model inference.

# Usage

MicroProphet is accessed via a command-line interface (CLI) with various modes. The general syntax is:
```bash
mph <mode> [options]
```

## Split data

**Input**: Abundance data in `csv` format which contains the following columns:
- **subject_id**: A unique identifier for each sample.
- **time**: The time point at which the sample was collected.
- **Columns for species** (e.g., `Actinobact`, `Alphaprot`, etc.): The relative abundance of each species in different samples (values are numerical).

Below is an example of how the data is structured:

| subject_id | time | Actinobact | Alphaprot | Bacilli | Bacteroidi | Betaprote | Clostridi | Cyanobact | Epsilonprc | ... |
|------------|------|------------|-----------|---------|------------|-----------|-----------|-----------|------------|-----|
| 41         | 1    | 0.0024976  | 0.000084  | 0.038825| 0          | 0         | 0.000282  | 0         | 0          | ... |
| 42         | 1    | 0.0023478  | 0.000076  | 0.038825| 0          | 0         | 0.000235  | 0         | 0          | ... |
| 43         | 1    | 0.0032016  | 0.000545  | 0       | 0          | 0         | 0.000193  | 0         | 0          | ... |
| 44         | 1    | 0.0026768  | 4.32E-06  | 0       | 0          | 0         | 0.000183  | 0         | 0          | ... |
| ...        | ...  | ...        | ...       | ...     | ...        | ...       | ...       | ...       | ...        | ... |

**Output**: The output consists of two datasets:
- **Training Set**: 70% of the subjects are used for training the model.
- **Test Set**: 30% of the subjects are used for testing the model.

These sets are divided based on the subject IDs to ensure that each subject is assigned either to the training or the test set. 


**Example:**

```bash
mph split ./data/invasion.csv --export_path="./data"
```

## Predict future microbial time-series abundance

- **Predict**: The model is trained on the time-series data of microbial abundance from the training set (70% of the subjects). The model learns the temporal dependencies between microbial abundance and time. Once trained, the model can forecast the future abundance of microbial species. This forecast is based on the time series and includes the next time points (e.g., next weeks, months, or other time intervals).

**Example:**

```bash
mph predict ./data/invasion.csv --export_path="./result" 
```

**Output**:
   The command will generate two main outputs:
   
   - **Predicted Microbial Abundance Table**:
     This table contains the predicted microbial abundances for the future time points, formatted similarly to the input data, but with time points continuing beyond the training data.

     Example of the predicted output:

     | subject_id | time | Actinobact (Predicted) | Alphaprot (Predicted) | Bacilli (Predicted) | ... |
     |------------|------|-----------------------|-----------------------|---------------------|-----|
     | 41          | 4    | 0.003152              | 0.000215              | 0.041278            | ... |
     | 42          | 4    | 0.003452              | 0.000175              | 0.039812            | ... |
     | 43          | 4    | 0.003829              | 0.000301              | 0.038613            | ... |

   - **Evaluation Metrics Table**:
     This table includes various evaluation metrics that assess the accuracy of the model's predictions, such as:
     - **Pearson Correlation Coefficient**: Measures the linear relationship between predicted and actual values.
     - **RMSE (Root Mean Squared Error)**: Measures the average magnitude of the prediction errors.
     
**Notes**:
- This model is useful for forecasting microbial dynamics over time, enabling better planning and management in microbiome studies or applications.

## Output attention heatmap

- **Attention Heatmap**: This heatmap visualizes how different time points influence each other in terms of microbial species abundance. The model calculates the attention scores to highlight the interaction between time points, showing how the abundance of species at one time point affects future time points.

**Example:**

```bash
mph attention ./data/invasion.csv --export_path="./result"
```

## Output SHAP Values for Species Importance

**Example:**

```bash
mph shap ./data/invasion.csv --export_path="./result"
```

- **SHAP Analysis**: SHAP values explain the contribution of each species to the model's predictions.

Example of the generated SHAP results table:

| Species        | SHAP Value (Importance) |
|----------------|-------------------------|
| Actinobact     | 0.0054                  |
| Alphaprot      | 0.0032                  |
| Bacilli        | 0.0015                  |
| Bacteroidi     | 0.0023                  |
| Betaprote      | 0.0009                  |
| ...            | ...                     |

# Dependencies 

## Microprophet Project Configuration

This document outlines the configuration for the **Microprophet** project, including the build system, project details, dependencies, and scripts.

## Build System

The project requires `setuptools` version `76.0.0` to build:

```toml
[build-system]
requires = ["setuptools==76.0.0"]

[project]
name = "microprophet"
version = "1.1.0"
description = "microprophet project"
readme = "README.md"
requires-python = ">=3.9.5,<3.10"
dependencies = [
    "accelerate>=1.5.2",
    "matplotlib>=3.9.4",
    "numba==0.59.1",
    "numpy==1.26.4",
    "pandas>=2.2.3",
    "shap>=0.46.0",
    "torch>=2.6.0",
    "transformers>=4.49.0",
]
```
# Others

## Parameter description

- **num_epoch**: The number of iterations to train the model. Controls how many times the entire dataset is processed during training.

- **lr**: The learning rate during training. It controls how quickly the model adapts to the data during training.

- **context_length**: The length of historical time steps used by the model to predict future values.

- **future_steps**: The number of future time points the model is expected to predict.

- **nhead**: The number of attention heads in the multi-head attention mechanism of the model, used in transformer-based architectures.

- **train_batch_size**: The number of samples processed per batch during training.

- **num_train_epochs**: The number of epochs (iterations over the entire dataset) during training.

- **evaluation_strategy**: Defines when to evaluate the model during training, e.g., after each epoch.

- **early_stopping_patience**: The number of epochs with no improvement in evaluation results before stopping the training to avoid overfitting.


# Maintainer

| Name | Email | Organization |
|-------|-------|-------|
| Yuli Zhang | yulizhang@hust.edu.cn | PhD student, School of Life Science and Technology, Huazhong University of Science & Technology |
| Kouyi Zhou | zhoukouyi@hust.edu.cn | PhD student, School of Life Science and Technology, Huazhong University of Science & Technology |
| Xiaoke Chen | cxk@hust.edu.cn | MD student, School of Life Science and Technology, Huazhong University of Science & Technology |
| Kang Ning  | ningkang@hust.edu.cn | Professor, School of Life Science and Technology, Huazhong University of Science & Technology|
