from django.shortcuts import render, redirect
from django.contrib import messages
from django.core.paginator import Paginator
from django.http import HttpResponse
from .forms import PredictionForm, BulkUploadForm, RetrainForm
from .models import PredictionHistory
from services.ml_service import ml_service
import pandas as pd
import io
from django.contrib.auth.decorators import login_required

from django.db.models import Count, Avg

def index(request):
    accuracy = ml_service.get_model_accuracy()
    return render(request, 'predictor/index.html', {'model_accuracy': accuracy})

def dashboard(request):
    total_predictions = PredictionHistory.objects.count()
    positive_responses = PredictionHistory.objects.filter(prediction_response=1).count()
    
    # Calculate response rate safely
    response_rate = (positive_responses / total_predictions * 100) if total_predictions > 0 else 0
    
    # Calculate average confidence
    avg_confidence = PredictionHistory.objects.aggregate(Avg('confidence_score'))['confidence_score__avg']
    avg_confidence = (avg_confidence * 100) if avg_confidence else 0
    
    recent_predictions = PredictionHistory.objects.order_by('-timestamp')[:5]
    
    accuracy = ml_service.get_model_accuracy()
    
    context = {
        'total_predictions': total_predictions,
        'positive_responses': positive_responses,
        'response_rate': response_rate,
        'avg_confidence': avg_confidence,
        'recent_predictions': recent_predictions,
        'model_accuracy': accuracy
    }
    return render(request, 'predictor/dashboard.html', context)

def single_prediction(request):
    if request.method == 'POST':
        form = PredictionForm(request.POST)
        if form.is_valid():
            feature_dict = form.cleaned_data
            
            try:
                prediction, confidence = ml_service.predict(feature_dict)
                
                history_record = form.save(commit=False)
                history_record.prediction_response = prediction
                history_record.confidence_score = confidence
                history_record.save()
                
                result_text = "Will Respond" if prediction == 1 else "Will Not Respond"
                messages.success(request, f'Prediction: {result_text} (Confidence: {confidence:.2f})')
                return redirect('single_prediction')
            except Exception as e:
                messages.error(request, f'Error during prediction: {e}')
    else:
        form = PredictionForm()
        
    return render(request, 'predictor/single_prediction.html', {'form': form})

def bulk_prediction(request):
    if request.method == 'POST':
        form = BulkUploadForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = request.FILES['csv_file']
            
            if not csv_file.name.endswith('.csv'):
                messages.error(request, 'Invalid file format. Please upload a CSV.')
                return redirect('bulk_prediction')
                
            try:
                df = pd.read_csv(csv_file)
                predictions = []
                confidences = []
                
                # Get the correct column order expected by the model
                features = ml_service.get_features_order()
                
                db_field_mapping = {
                    'Income': 'income', 'Kidhome': 'kidhome', 'Teenhome': 'teenhome', 'Recency': 'recency',
                    'MntWines': 'mnt_wines', 'MntFruits': 'mnt_fruits', 'MntMeatProducts': 'mnt_meat_products',
                    'MntFishProducts': 'mnt_fish_products', 'MntSweetProducts': 'mnt_sweet_products',
                    'MntGoldProds': 'mnt_gold_prods', 'NumDealsPurchases': 'num_deals_purchases',
                    'NumWebPurchases': 'num_web_purchases', 'NumCatalogPurchases': 'num_catalog_purchases',
                    'NumStorePurchases': 'num_store_purchases', 'NumWebVisitsMonth': 'num_web_visits_month',
                    'AcceptedCmp3': 'accepted_cmp3', 'AcceptedCmp4': 'accepted_cmp4', 'AcceptedCmp5': 'accepted_cmp5',
                    'AcceptedCmp1': 'accepted_cmp1', 'AcceptedCmp2': 'accepted_cmp2', 'Complain': 'complain',
                    'Age': 'age', 'Education_Encoded': 'education_encoded', 'Marital_Absurd': 'marital_absurd',
                    'Marital_Alone': 'marital_alone', 'Marital_Divorced': 'marital_divorced', 'Marital_Married': 'marital_married',
                    'Marital_Single': 'marital_single', 'Marital_Together': 'marital_together', 'Marital_Widow': 'marital_widow',
                    'Marital_YOLO': 'marital_yolo'
                }
                
                for _, row in df.iterrows():
                    feature_dict = {f: row.get(f, 0) for f in features}
                    pred, conf = ml_service.predict(feature_dict)
                    predictions.append(pred)
                    confidences.append(conf)
                    
                    db_dict = {db_field_mapping.get(f, f.lower()): val for f, val in feature_dict.items()}
                    
                    # Optional: save to history
                    PredictionHistory.objects.create(
                        **db_dict,
                        prediction_response=pred,
                        confidence_score=conf
                    )
                    
                df['Prediction'] = predictions
                df['Confidence'] = confidences
                
                response = HttpResponse(content_type='text/csv')
                response['Content-Disposition'] = 'attachment; filename="bulk_predictions.csv"'
                df.to_csv(path_or_buf=response, index=False)
                return response
                
            except Exception as e:
                messages.error(request, f'Error processing file: {e}')
    else:
        form = BulkUploadForm()
        
    return render(request, 'predictor/bulk_prediction.html', {'form': form})

import csv

def export_history_csv(request):
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="prediction_history.csv"'
    
    writer = csv.writer(response)
    # Write header
    writer.writerow([
        'ID', 'Timestamp', 'Prediction Response', 'Confidence Score',
        'Income', 'Age', 'Kidhome', 'Teenhome', 'Recency',
        'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts',
        'MntSweetProducts', 'MntGoldProds', 'NumDealsPurchases', 'NumWebPurchases',
        'NumCatalogPurchases', 'NumStorePurchases', 'NumWebVisitsMonth',
        'AcceptedCmp1', 'AcceptedCmp2', 'AcceptedCmp3', 'AcceptedCmp4', 'AcceptedCmp5',
        'Complain', 'Education_Encoded', 'Marital_Absurd', 'Marital_Alone',
        'Marital_Divorced', 'Marital_Married', 'Marital_Single', 'Marital_Together',
        'Marital_Widow', 'Marital_YOLO'
    ])
    
    # Write data
    records = PredictionHistory.objects.all().order_by('-timestamp')
    for record in records:
        writer.writerow([
            record.id, record.timestamp, record.prediction_response, record.confidence_score,
            record.income, record.age, record.kidhome, record.teenhome, record.recency,
            record.mnt_wines, record.mnt_fruits, record.mnt_meat_products, record.mnt_fish_products,
            record.mnt_sweet_products, record.mnt_gold_prods, record.num_deals_purchases, record.num_web_purchases,
            record.num_catalog_purchases, record.num_store_purchases, record.num_web_visits_month,
            record.accepted_cmp1, record.accepted_cmp2, record.accepted_cmp3, record.accepted_cmp4, record.accepted_cmp5,
            record.complain, record.education_encoded, record.marital_absurd, record.marital_alone,
            record.marital_divorced, record.marital_married, record.marital_single, record.marital_together,
            record.marital_widow, record.marital_yolo
        ])
        
    return response

def history_dashboard(request):
    history_list = PredictionHistory.objects.all().order_by('-timestamp')
    
    # Filter by prediction
    pred_filter = request.GET.get('prediction')
    if pred_filter in ['0', '1']:
        history_list = history_list.filter(prediction_response=int(pred_filter))
        
    paginator = Paginator(history_list, 20)
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)
    
    return render(request, 'predictor/history.html', {'page_obj': page_obj, 'pred_filter': pred_filter})

def retrain_model(request):
    if request.method == 'POST':
        form = RetrainForm(request.POST, request.FILES)
        if form.is_valid():
            csv_file = request.FILES['csv_file']
            
            if not csv_file.name.endswith('.csv'):
                messages.error(request, 'Invalid file format. Please upload a CSV.')
                return redirect('retrain_model')
                
            try:
                # Save uploaded file temporarily
                temp_path = 'temp_retrain.csv'
                with open(temp_path, 'wb+') as f:
                    for chunk in csv_file.chunks():
                        f.write(chunk)
                        
                ml_service.retrain_model(temp_path)
                
                import os
                os.remove(temp_path)
                
                messages.success(request, 'Model retrained successfully!')
                return redirect('retrain_model')
            except Exception as e:
                messages.error(request, f'Error during retraining: {e}')
    else:
        form = RetrainForm()
        
    return render(request, 'predictor/retrain.html', {'form': form})

def model_accuracy(request):
    import os
    from django.conf import settings
    csv_path = os.path.join(settings.BASE_DIR, 'Model-Dataset', 'ml', 'model_comparison_results.csv')
    try:
        df = pd.read_csv(csv_path)
        df.columns = [c.replace('-', '_') for c in df.columns]
        
        dynamic_acc = ml_service.get_model_accuracy() / 100.0
        for idx, row in df.iterrows():
            if row['Model'] == 'XGBoost' and row['Dataset'] == 'Scaled':
                df.at[idx, 'Accuracy'] = dynamic_acc
                
        records = df.to_dict('records')
        
        # Prepare Chart Data
        chart_data = {
            'labels': ['Logistic Regression', 'Random Forest', 'Gradient Boosting', 'SVM', 'XGBoost'],
            'datasets': [
                {'label': 'Scaled Data', 'data': [], 'backgroundColor': 'rgba(30, 60, 114, 0.85)'},
                {'label': 'Raw Data', 'data': [], 'backgroundColor': 'rgba(42, 190, 150, 0.85)'},
                {'label': 'PCA Data', 'data': [], 'backgroundColor': 'rgba(255, 193, 7, 0.85)'}
            ]
        }
        
        for ds in chart_data['datasets']:
            ds_name = ds['label'].replace(' Data', '')
            for model in chart_data['labels']:
                val = 0
                for row in records:
                    if row['Model'] == model and row['Dataset'] == ds_name:
                        val = round(row['Accuracy'] * 100, 2)
                        break
                ds['data'].append(val)
                
        import json
        chart_json = json.dumps(chart_data)
        
    except Exception as e:
        records = []
        chart_json = "{}"
        messages.error(request, f"Could not load accuracy data: {e}")
        
    return render(request, 'predictor/accuracy.html', {'records': records, 'chart_json': chart_json})

def visualize_model(request):
    import json
    feature_importances = ml_service.get_feature_importances()
    
    # Extract labels and data for Chart.js
    labels = [item['feature'] for item in feature_importances]
    data = [item['importance'] for item in feature_importances]
    
    context = {
        'chart_labels': json.dumps(labels),
        'chart_data': json.dumps(data)
    }
    return render(request, 'predictor/visualize.html', context)

def dataset_strategies(request):
    import os
    import json
    import pandas as pd
    from django.conf import settings
    
    csv_path = os.path.join(settings.BASE_DIR, 'Model-Dataset', 'ml', 'model_comparison_results.csv')
    try:
        df = pd.read_csv(csv_path)
        # Calculate average accuracy per dataset type
        avg_acc = df.groupby('Dataset')['Accuracy'].mean() * 100
        
        chart_data = {
            'labels': ['Raw Data Baseline', 'Scaled Data (Standardization)', 'PCA Data (Dimensionality Reduction)'],
            'datasets': [{
                'label': 'Average Model Accuracy (%)',
                'data': [
                    round(avg_acc.get('Raw', 0), 2),
                    round(avg_acc.get('Scaled', 0), 2),
                    round(avg_acc.get('PCA', 0), 2)
                ],
                'backgroundColor': [
                    'rgba(42, 190, 150, 0.85)',  # Teal for Raw
                    'rgba(30, 60, 114, 0.85)',   # Blue for Scaled
                    'rgba(255, 193, 7, 0.85)'    # Yellow for PCA
                ],
                'borderWidth': 1
            }]
        }
        chart_json = json.dumps(chart_data)
    except Exception as e:
        chart_json = "{}"
        
    return render(request, 'predictor/datasets.html', {'chart_json': chart_json})
