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

Note: This project can also be run in a Google Colab environment, which typically has the required dependencies pre-installed and includes CUDA support. You will need to update file paths at the top of each notebook if the dataset files are located elsewhere.

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

### 3. Data Preprocessing

Run the preprocessing notebook to prepare the data:

```bash
# Navigate to preprocessing/data_preprocessing.ipynb and run all cells
```

This notebook will:
- Download the fraud dataset csv file ([Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud/data?select=creditcard.csv))
- Load and explore the credit card fraud dataset
- Perform data cleaning and feature engineering
- Split data into training and testing sets
- Save preprocessed data for model training

### 4. FT-Transformer Implementation

Run the main FT-Transformer notebook:

```bash
# In Jupyter, open notebooks/FTtransformer_fraud_classifier.ipynb
```

This notebook includes:
- FT-Transformer model implementation
- Model training and evaluation
- SHAP explanations for model interpretability
- Performance metrics and visualizations

### 5. Traditional Model Benchmarks

Compare FT-Transformer performance with traditional ML models:

```bash
# In Jupyter, open notebooks/traditional_model_benchmarks.ipynb
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
# In Jupyter, open notebooks/FTtransformer_fraud_classifier_grid_search.ipynb
```

This notebook performs:
- Grid search over key hyperparameters
- Cross-validation for robust evaluation
- Best parameter selection

## Usage Notes

- **Data**: The project uses the credit card fraud dataset (`creditcard.csv`) located in the `data/` directory
- **Environment**: Make sure to activate the virtual environment before running any notebooks
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