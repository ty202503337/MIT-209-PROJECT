import joblib
import pandas as pd
import warnings
from sklearn.metrics import accuracy_score, classification_report

warnings.filterwarnings('ignore')

model_path = 'Model-Dataset/models/xgb_marketing_model.pkl'
data_path = 'Model-Dataset/datasetsversion/xgb-lgb_marketing_data.csv'

try:
    # 1. Load the model
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    # 2. Load the dataset
    print(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Identify target column
    target_col = None
    for col in ['Response', 'response', 'Prediction Response', 'Prediction_Response']:
        if col in df.columns:
            target_col = col
            break
            
    if not target_col:
        print(f"Error: Target column not found. Available columns: {list(df.columns)}")
        exit()
        
    feature_names = model.feature_names_in_
    X = df[feature_names]
    y_true = df[target_col]
    
    print("Generating predictions...")
    y_pred = model.predict(X)
    
    accuracy = accuracy_score(y_true, y_pred)
    acc_percent = round(accuracy * 100, 2)
    
    # Save the accuracy to current_metrics.json to sync with web dashboard
    import json
    import os
    metrics_path = 'Model-Dataset/models/current_metrics.json'
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump({'accuracy': acc_percent}, f)
        
    print("\n" + "="*40)
    print(" MODEL ACCURACY REPORT")
    print("="*40)
    print(f"Overall Accuracy: {acc_percent}%\n")
    print("Detailed Classification Report:")
    print(classification_report(y_true, y_pred))
    
except Exception as e:
    print("Error:", e)
