# Fraud Detection with FT-Transformer

This project implements fraud detection using FT-Transformer (Feature Tokenizer Transformer) along with traditional machine learning benchmarks and SHAP explanations.

## Project Structure

```
Fraud-FT-Transformer/
├── data/
│                                                   # Folder to store datasets
├── notebooks/
│   ├── FT-Transformer_grid_search.ipynb            # Grid search optimization
│   ├── FT-Transformer.ipynb                        # Main FT-Transformer with SHAP
│   └── Traditional_model_benchmarks.ipynb          # Benchmark models comparison
├── preprocessing/
│   └── data_preprocessing.ipynb                    # Data preprocessing pipeline
├── results/                                        # Model results and outputs
└── requirements.txt                                # Python dependencies
```
## Dataset Used

This project uses the Credit Card Fraud Detection dataset from Kaggle:
- **Dataset**: [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data?select=creditcard.csv)
- **Description**: Contains transactions made by credit cards in September 2013 by European cardholders
- **Features**: 28 anonymized features (V1-V28) plus Time and Amount
- **Target**: Binary classification (0: Normal, 1: Fraud)


## Research Environment

To closely reproduce the results from the research paper, use the following exact environment configuration:

### Hardware Specifications
- **GPU**: NVIDIA L4 (22.2 GB memory, Compute capability: 8.9)
- **CUDA**: Version 12.6
- **Platform**: Linux-6.6.105+-x86_64-with-glibc2.35

### Software Versions
- **Python**: 3.12.12 (main, Oct 10 2025, 08:52:57) [GCC 11.4.0]
- **PyTorch**: 2.8.0+cu126
- **NumPy**: 2.0.2
- **Pandas**: 2.2.2
- **scikit-learn**: 1.6.1
- **rtdl-revisiting-models**: 0.0.2
- **CUDA**: 12.6
- **CUDNN**: 91002

> **Note**: Using different hardware or library versions may result in slightly different results. For exact reproducibility, match the environment specifications above.

## Setup Instructions

### 1. Create Virtual Environment

Create a Python virtual environment to isolate project dependencies:

```bash
# Create virtual environment
python3.12 -m venv fraud_detection_env

# Activate virtual environment
# On Linux/Mac:
source fraud_detection_env/bin/activate
# On Windows:
fraud_detection_env\Scripts\activate
```

### 2. Install Dependencies

Install the required packages from the requirements file:

```bash
pip install -r requirements.txt
```

> **Note**: For CUDA-enabled PyTorch, install the appropriate version manually if needed:
> ```bash
> pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
> ```

## Running the Project

### 3. Data Preprocessing

Run the preprocessing notebook to prepare the data:

```bash
# Navigate to preprocessing/data_preprocessing.ipynb and run all cells
```

This notebook will:
- Load and explore the credit card fraud dataset
- Perform data cleaning and feature engineering
- Split data into training and testing sets
- Save preprocessed data for model training

### 4. FT-Transformer Implementation

Run the main FT-Transformer notebook:

```bash
# In Jupyter, open notebooks/FT-Transformer.ipynb
```

This notebook includes:
- FT-Transformer model implementation
- Model training and evaluation
- SHAP explanations for model interpretability
- Performance metrics and visualizations

### 5. Traditional Model Benchmarks

Compare FT-Transformer performance with traditional ML models:

```bash
# In Jupyter, open notebooks/Traditional_model_benchmarks.ipynb
```

This notebook benchmarks:
- Logistic Regression
- Random Forest
- XGBoost
- Support Vector Machine
- Neural Networks

### 6. Grid Search Optimization

Run hyperparameter optimization for FT-Transformer:

```bash
# In Jupyter, open notebooks/FT-Transformer_grid_search.ipynb
```

This notebook performs:
- Grid search over key hyperparameters
- Cross-validation for robust evaluation
- Best parameter selection

## Notebook Execution Order

**Important**: The notebooks must be run in the following order:

1. **First**: `preprocessing/data_preprocessing.ipynb` - Generates the required `train.csv`, `test.csv`, and `preprocessing_metadata.json` files
2. **Then**: Any of the following notebooks (they depend on the preprocessed files):
   - `notebooks/FT-Transformer.ipynb` - Main FT-Transformer with SHAP analysis
   - `notebooks/FT-Transformer_grid_search.ipynb` - Hyperparameter optimization
   - `notebooks/Traditional_model_benchmarks.ipynb` - Benchmark comparisons

## Usage Notes

- **Data Dependencies**: 
  - Initial dataset: `creditcard.csv` (downloaded from Kaggle)
  - Generated files: `train.csv`, `test.csv`, `preprocessing_metadata.json` (created by preprocessing notebook)
  - The FT-Transformer notebooks require all 3 generated files to run properly
- **Environment**: Make sure to activate the virtual environment before running any notebooks
- **Reproducibility**: For exact results matching the research paper, use the specified Python 3.12.12 and library versions
- **GPU Requirements**: NVIDIA GPU with CUDA 12.6 support recommended for training (CUDA-enabled PyTorch required)
- **Memory**: FT-Transformer may require significant memory for large datasets; consider reducing batch size if needed

## Key Features

- **FT-Transformer**: State-of-the-art transformer architecture for tabular data
- **SHAP Integration**: Model interpretability and feature importance analysis
- **Comprehensive Benchmarking**: Comparison with traditional ML approaches
- **Hyperparameter Optimization**: Automated grid search for optimal performance
- **Visualization**: Rich plots and metrics for model evaluation

## Troubleshooting

- If you encounter memory issues, try reducing batch size in the model configuration
- Ensure all dependencies are installed correctly using `pip list`
- Check that the virtual environment is activated before running notebooks
- Verify that the credit card dataset is present in the `data/` directory