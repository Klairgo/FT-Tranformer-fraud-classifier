# Fraud Detection with FT-Transformer

This project implements fraud detection using FT-Transformer (Feature Tokenizer Transformer) along with traditional machine learning benchmarks and SHAP explanations.

## Project Structure

```
Fraud-FT-Transformer/
├── data/
│   └── creditcard.csv                              # Credit card fraud dataset
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


## Setup Instructions

### Option 1: Google Colab (Recommended)

> **_Note:_** Google Colab is the recommended environment as it provides pre-installed dependencies and CUDA support for faster training.

#### Step 1: Run Data Preprocessing
1. Open and run `preprocessing/data_preprocessing.ipynb` in Google Colab
2. The notebook will automatically download the dataset from Google Drive
3. If the automatic download fails, manually download the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data?select=creditcard.csv) (`creditcard.csv`) and upload it to your Colab environment
4. This will generate 3 essential files in the `/content/data/` directory:
   - `train.csv` - Training dataset
   - `test.csv` - Testing dataset  
   - `preprocessing_metadata.json` - Preprocessing parameters and feature information

#### Step 2: Save Preprocessed Files
**Important**: Download and save the 3 generated files (`train.csv`, `test.csv`, `preprocessing_metadata.json`) from the Colab environment, as they will be needed for the FT-Transformer notebooks.

#### Step 3: Run FT-Transformer Notebooks
1. For each subsequent notebook (`FT-Transformer.ipynb`, `FT-Transformer_grid_search.ipynb`):
   - Upload the 3 preprocessed files (`train.csv`, `test.csv`, `preprocessing_metadata.json`) to the Colab session
   - Update file paths in the notebook if needed (typically in the first few cells)
   - Run all cells

### Option 2: Local Setup

**The steps below are for local development setup.**

### 1. Create Virtual Environment

Create a Python virtual environment to isolate project dependencies:

```bash
# Create virtual environment
python -m venv fraud_detection_env

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

## Running the Project

### For Local Setup Only

#### 3. Data Preprocessing

Run the preprocessing notebook to prepare the data:

```bash
# Navigate to preprocessing/data_preprocessing.ipynb and run all cells
```

This notebook will:
- Load and explore the credit card fraud dataset
- Perform data cleaning and feature engineering
- Split data into training and testing sets
- Save preprocessed data for model training

#### 4. FT-Transformer Implementation

Run the main FT-Transformer notebook:

```bash
# In Jupyter, open notebooks/FT-Transformer.ipynb
```

This notebook includes:
- FT-Transformer model implementation
- Model training and evaluation
- SHAP explanations for model interpretability
- Performance metrics and visualizations

#### 5. Traditional Model Benchmarks

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

#### 6. Grid Search Optimization

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
- **Environment**: Make sure to activate the virtual environment before running any notebooks (local setup only)
- **Google Colab**: File paths may need adjustment in notebook cells when switching between Colab sessions
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