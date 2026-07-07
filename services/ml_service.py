import os
import joblib
import pandas as pd
import xgboost as xgb
from django.conf import settings

# Path to the model
MODEL_PATH = os.path.join(settings.BASE_DIR, 'Model-Dataset', 'models', 'xgb_marketing_model.pkl')

class MLService:
    def __init__(self):
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            self.model = joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"Error loading model: {e}")
            raise RuntimeError(f"Could not load ML model: {e}")

    def get_features_order(self):
        return [
            'Income', 'Kidhome', 'Teenhome', 'Recency', 'MntWines', 'MntFruits', 
            'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 
            'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases', 'NumStorePurchases', 
            'NumWebVisitsMonth', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5', 
            'AcceptedCmp1', 'AcceptedCmp2', 'Complain', 'Age', 'Education_Encoded', 
            'Marital_Absurd', 'Marital_Alone', 'Marital_Divorced', 'Marital_Married', 
            'Marital_Single', 'Marital_Together', 'Marital_Widow', 'Marital_YOLO'
        ]

    def predict(self, feature_dict):
        if self.model is None:
            self.load_model()
            
        features_order = self.get_features_order()
        mapped_dict = {}
        
        for f in features_order:
            if f in feature_dict:
                mapped_dict[f] = feature_dict[f]
            else:
                # Find matching key ignoring case and underscores
                target = f.lower().replace('_', '')
                for k, v in feature_dict.items():
                    if k.lower().replace('_', '') == target:
                        mapped_dict[f] = v
                        break
                else:
                    mapped_dict[f] = 0
                    
        df = pd.DataFrame([mapped_dict])
        df = df[features_order]
        
        prediction = self.model.predict(df)[0]
        try:
            proba = self.model.predict_proba(df)[0]
            confidence = proba[1] if prediction == 1 else proba[0]
        except AttributeError:
            confidence = 1.0
            
        return int(prediction), float(confidence)

    def get_feature_importances(self):
        if self.model is None:
            self.load_model()
            
        try:
            importances = self.model.feature_importances_
            features = self.model.feature_names_in_
            
            # Combine, sort, and get top 15 features
            feat_imp = list(zip(features, importances))
            feat_imp.sort(key=lambda x: x[1], reverse=True)
            return [{'feature': f, 'importance': round(float(imp) * 100, 2)} for f, imp in feat_imp[:15]]
        except Exception as e:
            print("Error extracting feature importances:", e)
            return []
    def get_model_accuracy(self):
        import json
        metrics_path = os.path.join(settings.BASE_DIR, 'Model-Dataset', 'models', 'current_metrics.json')
        if os.path.exists(metrics_path):
            try:
                with open(metrics_path, 'r') as f:
                    data = json.load(f)
                    return round(data.get('accuracy', 97.14), 2)
            except Exception:
                pass
        return 97.14

    def retrain_model(self, csv_file_path):
        df = pd.read_csv(csv_file_path)
        
        # Support various target column names
        target_col = None
        for col in ['Response', 'response', 'Prediction Response', 'Prediction_Response']:
            if col in df.columns:
                target_col = col
                break
                
        if target_col is None:
            raise KeyError(f"Target column 'Response' not found. Available columns: {list(df.columns)}")
            
        X = df[self.get_features_order()]
        y = df[target_col]
        
        # Fit model
        self.model.fit(X, y)
        
        # Calculate and save new accuracy
        from sklearn.metrics import accuracy_score
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred) * 100
        
        import json
        metrics_path = os.path.join(settings.BASE_DIR, 'Model-Dataset', 'models', 'current_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump({'accuracy': accuracy}, f)
        
        # Save model
        joblib.dump(self.model, MODEL_PATH)
        return True

ml_service = MLService()
