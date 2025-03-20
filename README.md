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
| 41         | 3    | 0.0024976  | 0.000084  | 0.038825| 0          | 0         | 0.000282  | 0         | 0          | ... |
| 42         | 3    | 0.0023478  | 0.000076  | 0.038825| 0          | 0         | 0.000235  | 0         | 0          | ... |
| 43         | 3    | 0.0032016  | 0.000545  | 0       | 0          | 0         | 0.000193  | 0         | 0          | ... |
| 44         | 3    | 0.0026768  | 4.32E-06  | 0       | 0          | 0         | 0.000183  | 0         | 0          | ... |
| ...        | ...  | ...        | ...       | ...     | ...        | ...       | ...       | ...       | ...        | ... |

**Output**: The output consists of two datasets:
- **Training Set**: 70% of the subjects are used for training the model.
- **Test Set**: 30% of the subjects are used for testing the model.

These sets are divided based on the subject IDs to ensure that each subject is assigned either to the training or the test set. 

**Example:**

```bash
mph split ./data/invasion.csv --export_path="./data"
```



## Download models

# Maintainer

| Name | Email | Organization |
|-------|-------|-------|
| Yuli Zhang | yulizhang@hust.edu.cn | PhD student, School of Life Science and Technology, Huazhong University of Science & Technology |
| Kouyi Zhou | zhoukouyi@hust.edu.cn | PhD student, School of Life Science and Technology, Huazhong University of Science & Technology |
| Kang Ning  | ningkang@hust.edu.cn | Professor, School of Life Science and Technology, Huazhong University of Science & Technology|
