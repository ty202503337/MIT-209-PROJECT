from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('predict/', views.single_prediction, name='single_prediction'),
    path('bulk/', views.bulk_prediction, name='bulk_prediction'),
    path('history/', views.history_dashboard, name='history'),
    path('history/export/', views.export_history_csv, name='export_history'),
    path('retrain/', views.retrain_model, name='retrain_model'),
    path('accuracy/', views.model_accuracy, name='model_accuracy'),
    path('visualize/', views.visualize_model, name='visualize_model'),
    path('datasets/', views.dataset_strategies, name='datasets'),
]
