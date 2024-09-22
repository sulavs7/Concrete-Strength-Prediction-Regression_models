# Concrete Strength Prediction

This project aims to predict the compressive strength of concrete using three different regression models: Random Forest, Linear Regression, and Decision Tree Regression. It allows users to input their dataset and receive predictions based on various features.
## Features
- Predicts concrete strength based on multiple features.

- Includes pre-trained models for Random Forest, Linear Regression, and Decision Tree Regression.

- Users can input their own datasets in CSV format for predictions.

## Files in the Repository

- `src/main.py`                      # Main module where users can input their dataset and check model predictions.
- `src/evaluate.py`                  # A script that calculates evaluation metrics for the models.
- `src/concrete.ipynb`               # Jupyter Notebook for model training.
- `models/random_forest_model.pkl`    # Pre-trained Random Forest model file.
- `models/linear_regression_model.pkl`  # Pre-trained Linear Regression model file.
- `models/decision_tree_model.pkl`     # Pre-trained Decision Tree model file.
- `data/validation_data.csv`           # Example dataset for validating the models.
- `data/predictions.csv`               # CSV file where predicted values are saved.
- `requirements.txt`                   # List of required dependencies for the project.

## Requirements
To run this project, you need to have the following libraries installed:

```txt
pandas==1.5.3
scikit-learn==1.2.0
joblib==1.2.0
numpy==1.24.2
```

## You can install these using the requirements.txt file by following command mentioned bellow in your terminal:
```txt
pip install -r requirements.txt
```
## How to Clone the Repository
To clone this repository, run the following command in your terminal:
```txt
git clone https://github.com/sulavs7/Concrete-Strength-Prediction.git

```

Once cloned, navigate to the project directory:
```txt
cd Concrete-Strength-Prediction
```

## How to Use
1.Make sure the required packages are installed (pandas, scikit-learn, etc.).

2.Run the main.py script in the src folder.

3.Enter the path to your dataset when prompted. The predictions will be saved to predictions.csv.

## License
This project is licensed under the MIT License.

