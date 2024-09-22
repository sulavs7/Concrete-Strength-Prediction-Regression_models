import numpy as np 
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_prediction(y,y_pred):
    mse=mean_squared_error(y,y_pred)
    r2=r2_score(y,y_pred)

    metrics={
        "mse":mse,
        "r2":r2
    }
    return metrics

if __name__=="__main__":
    y=[3,4,5,6,7,8,2,1,3]
    y_pred=[3,4,5,6,7,8,2,1,3]

    metrics = evaluate_prediction(y,y_pred)
    for metric, value in metrics.items():
            print(f"{metric}:\n{value}")
    
    


    

