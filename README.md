# healthcare-imputation-evaluation

This project provides a command-line tool for handling missing data in time-series and clinical datasets. It offers multiple modes of operation, including data imputation using a range of techniques, synthetic missing data creation for simulation and testing, and evaluation of imputation performance through visual comparisons.

## Features

- **Imputation Mode:**  
  Apply different imputation methods (KNN, MICE, MICE Forest, GRU-RNN, and Self-Supervised Learning) on observed datasets.
  
- **Missing Data Creation Mode:**  
  Generate datasets with missing values from a full dataset. This helps simulate real-world missingness patterns and test imputation quality.
  
- **Evaluation Mode:**  
  Compare imputed data with ground truth via evaluation metrics and line plots. The tool handles different datasets (e.g., MIMICIV, MINIC) and adapts to their specific configurations.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- Required libraries:  
  -  pandas  
  -  argparse
  -  keras
  -  matplotlib
  -  miceforest
  -  numpy
  -  pandas
  -  scikit-learn
  -  seaborn
  -  tensorflow
  -  tqdm

It is assumed that you have a proper virtual environment setup and the required packages installed (for example, via a requirements.txt file).

### Installation

1. Clone the repository:  
   `git clone https://github.com/NIMI-research/healthcare-imputation-evaluation.git`
   
2. Navigate into the repository directory:  
   `cd healthcare-imputation-evaluation`
   
3. Install the required packages:  
   `pip install -r requirements.txt`

## Usage

The main script uses the Python `argparse` module to enable three different modes of operation. Run the script from the command line with one of the following flags:

### Imputation Mode

To perform data imputation:
```
python main.py --impute
```
In this mode, the script:
- Loads the configuration and dataset for the chosen dataset (MIMICIV, or MINIC).
- Uses the selected imputation technique. For methods such as KNN, MICE, and Miceforest, a direct function call is made. For GRU (nn) and Self-Supervised Learning (ssl) methods, the tool either trains, transforms, or both based on sub-configurations.

### Missing Data Creation Mode

To generate a dataset with missing values:
```
python main.py --create_missing
```
This mode:
- Loads the full dataset.
- Applies missing data creation routines (via the `create_missing_dataset` function) using configurations such as the percentage of data to remove and the columns to target.

### Evaluation Mode

To evaluate and visualize the performance of imputations:
```
python main.py --eval
```
In evaluation mode, the script:
- Loads the evaluation columns from the dataset.
- Maps column names to labels (using dictionaries for MIMICIV and MINIC datasets).
- Plots a line chart comparing the ground truth with the imputed values (using the `plot_line` function).

## Configuration

Several helper functions are used to load configuration details and dataset-specific settings:
- `get_config("impute")`, `get_config("create_missing")`, and `get_config("eval")` obtain configurations (e.g., dataset name, imputer type, missing value details) from YAML or other configuration files.
- Dataset configurations are defined in classes such as `MimicConfig`, `MinicConfig`, and `HopperConfig` located in the `dataset_config` module.

Make sure your YAML configuration files and your dataset-specific settings are correctly set up for your intended use case.

## Code Structure

- **Main Script:**  
  Contains the entry point (`if __name__ == "__main__":`) that parses command-line arguments and directs the program flow to either imputation, missing data creation, or evaluation.

- **Utils Directory:**  
  -  `load_config.py`: Handles configuration fetching (YAML and other formats).  
  -  `load_dataset.py`: Provides helpers to load observational, full, or evaluation datasets.  
  -  `create_missing_data.py`: Contains routines to generate missing data in the dataset.  
  -  `plot_predictions.py`: Includes plotting functions for evaluation output.

- **Imputers Directory:**  
  Contains modules implementing various imputation strategies:
  -  `knn.py`  
  -  `mice.py`  
  -  `nnimputer.py`  
  -  `ssl_imputer.py`

 ## Datasets

 ### MIMIC-IV Dataset

The **MIMIC-IV** dataset is a publicly available database containing de-identified health records from patients admitted to intensive care units (ICUs) at Beth Israel Deaconess Medical Center. It includes both structured clinical data and unstructured notes.

### Accessing MIMIC-IV Data

1. Visit the PhysioNet website.
2. Complete the credentialing process if you are a new user.
4. Follow detailed access instructions at [PhysioNet's MIMIC-IV documentation](https://mimic-iv.mit.edu/docs/access/).

### Preprocessing Steps

The raw MIMIC-IV dataset requires preprocessing before use:
- **Data Cleaning:** Handle inconsistent units by converting to standard formats, remove outliers, and resolve duplicate entries by averaging numerical values or selecting the first categorical value.
- **Feature Selection:** Select relevant features for analysis, such as demographic information, admission details, and clinical measurements.
- **Imputation Preparation:** Identify missing values and prepare datasets for imputation tasks.
