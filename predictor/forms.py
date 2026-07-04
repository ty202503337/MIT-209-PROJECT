from django import forms
from .models import PredictionHistory

class PredictionForm(forms.ModelForm):
    class Meta:
        model = PredictionHistory
        exclude = ['timestamp', 'prediction_response', 'confidence_score']
        labels = {
            'income': 'Annual Income (₱)',
            'kidhome': 'Children in Home',
            'teenhome': 'Teens in Home',
            'recency': 'Days Since Last Purchase',
            'mnt_wines': 'Amount Spent on Wines (₱)',
            'mnt_fruits': 'Amount Spent on Fruits (₱)',
            'mnt_meat_products': 'Amount Spent on Meat (₱)',
            'mnt_fish_products': 'Amount Spent on Fish (₱)',
            'mnt_sweet_products': 'Amount Spent on Sweets (₱)',
            'mnt_gold_prods': 'Amount Spent on Gold (₱)',
            'num_deals_purchases': 'Purchases with Discount',
            'num_web_purchases': 'Web Purchases',
            'num_catalog_purchases': 'Catalog Purchases',
            'num_store_purchases': 'Store Purchases',
            'num_web_visits_month': 'Web Visits per Month',
            'accepted_cmp1': 'Accepted Campaign 1',
            'accepted_cmp2': 'Accepted Campaign 2',
            'accepted_cmp3': 'Accepted Campaign 3',
            'accepted_cmp4': 'Accepted Campaign 4',
            'accepted_cmp5': 'Accepted Campaign 5',
            'complain': 'Complained in Last 2 Yrs',
            'age': 'Age',
            'education_encoded': 'Education Level (Encoded)',
            'marital_absurd': 'Marital Status: Absurd',
            'marital_alone': 'Marital Status: Alone',
            'marital_divorced': 'Marital Status: Divorced',
            'marital_married': 'Marital Status: Married',
            'marital_single': 'Marital Status: Single',
            'marital_together': 'Marital Status: Together',
            'marital_widow': 'Marital Status: Widow',
            'marital_yolo': 'Marital Status: YOLO',
        }
        widgets = {
            'kidhome': forms.Select(choices=[(0, '0'), (1, '1'), (2, '2'), (3, '3')]),
            'teenhome': forms.Select(choices=[(0, '0'), (1, '1'), (2, '2'), (3, '3')]),
            'education_encoded': forms.Select(choices=[(0, 'Basic'), (1, '2n Cycle'), (2, 'Graduation'), (3, 'Master'), (4, 'PhD')]),
            'accepted_cmp1': forms.Select(choices=[(0, 'No'), (1, 'Yes')]),
            'accepted_cmp2': forms.Select(choices=[(0, 'No'), (1, 'Yes')]),
            'accepted_cmp3': forms.Select(choices=[(0, 'No'), (1, 'Yes')]),
            'accepted_cmp4': forms.Select(choices=[(0, 'No'), (1, 'Yes')]),
            'accepted_cmp5': forms.Select(choices=[(0, 'No'), (1, 'Yes')]),
            'complain': forms.Select(choices=[(0, 'No'), (1, 'Yes')]),
            'marital_absurd': forms.Select(choices=[(0, 'No'), (1, 'Yes')]),
            'marital_alone': forms.Select(choices=[(0, 'No'), (1, 'Yes')]),
            'marital_divorced': forms.Select(choices=[(0, 'No'), (1, 'Yes')]),
            'marital_married': forms.Select(choices=[(0, 'No'), (1, 'Yes')]),
            'marital_single': forms.Select(choices=[(0, 'No'), (1, 'Yes')]),
            'marital_together': forms.Select(choices=[(0, 'No'), (1, 'Yes')]),
            'marital_widow': forms.Select(choices=[(0, 'No'), (1, 'Yes')]),
            'marital_yolo': forms.Select(choices=[(0, 'No'), (1, 'Yes')]),
        }


class BulkUploadForm(forms.Form):
    csv_file = forms.FileField(label='Upload CSV Dataset', help_text='Upload a CSV matching the feature schema.')

class RetrainForm(forms.Form):
    csv_file = forms.FileField(label='Upload Retraining CSV', help_text='CSV must contain the target column "Response".')
