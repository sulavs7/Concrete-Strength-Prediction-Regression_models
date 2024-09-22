import joblib 
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

def load_model():
    models={
        "RandomForest":joblib.load("RandomForest.pkl"),
        "LinearRegression":joblib.load("LinearRegression.pkl"),
        "DecisionTreeRegressor":joblib.load("DecisionTreeRegressor.pkl")
    }
    return models
def choose_model(models):
    for idx,(key,value) in enumerate(models.items()):
        print(f"{idx}.{key}")
    user_choice=int(input("Enter The model you Want to use(2.has best score):")) #decision tree has best score
    # Checking if the choice is valid
    if user_choice < 0 or user_choice >= len(models):
        print("Invalid choice. Please select a valid model.")
        return None  # Handle invalid choice as needed

    selected_model = list(models.values())[user_choice]
    return selected_model

def preprocess_input(input_data):
    input_data.fillna(input_data.mean(),inplace=True)

    features=[feature for feature in input_data.columns]
    if "concrete_compressive_strength" in features:
        input_data=input_data.drop("concrete_compressive_strength",axis=1) #dropping target

    expected_columns=['cement', 'blast_furnace_slag', 'fly_ash', 'water', 'superplasticizer', 'coarse_aggregate', 'fine_aggregate ', 'age']

   
    if not all(col in features for col in expected_columns):
        raise ValueError("Input data must contain the following columns: " + ", ".join(expected_columns))

    # Scale features
    scaler = StandardScaler()  # Using the same scaler fitted on training data
    data_scaled = scaler.fit_transform(input_data[expected_columns])

    return pd.DataFrame(data_scaled, columns=expected_columns)  # Returning as DataFrame
    
def make_predictions(selected_model,processed_data):
    predictions = selected_model.predict(processed_data)
    return predictions


def main():
    models=load_model()
    selected_model=choose_model(models)
    print(selected_model)

    user_input_file = input("Enter the path to your dataset (CSV format): ")
    input_data = pd.read_csv(user_input_file)

     # Preprocess the input data
    processed_data = preprocess_input(input_data)

    predictions = make_predictions(selected_model, processed_data)
    predictions_df=pd.DataFrame({
        "concrete_compressive_strength":predictions
    })
    predictions_df.to_csv("prediction.csv",index=False)
    print("Predictions saved to prediction.csv!!\n Check 'prediction.csv' for predictions")


if __name__=="__main__":
    main()