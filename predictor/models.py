from django.db import models
from django.utils import timezone

class PredictionHistory(models.Model):
    timestamp = models.DateTimeField(default=timezone.now)
    income = models.FloatField(verbose_name="Income")
    kidhome = models.IntegerField(verbose_name="Kidhome")
    teenhome = models.IntegerField(verbose_name="Teenhome")
    recency = models.IntegerField(verbose_name="Recency")
    mnt_wines = models.FloatField(verbose_name="MntWines")
    mnt_fruits = models.FloatField(verbose_name="MntFruits")
    mnt_meat_products = models.FloatField(verbose_name="MntMeatProducts")
    mnt_fish_products = models.FloatField(verbose_name="MntFishProducts")
    mnt_sweet_products = models.FloatField(verbose_name="MntSweetProducts")
    mnt_gold_prods = models.FloatField(verbose_name="MntGoldProds")
    num_deals_purchases = models.IntegerField(verbose_name="NumDealsPurchases")
    num_web_purchases = models.IntegerField(verbose_name="NumWebPurchases")
    num_catalog_purchases = models.IntegerField(verbose_name="NumCatalogPurchases")
    num_store_purchases = models.IntegerField(verbose_name="NumStorePurchases")
    num_web_visits_month = models.IntegerField(verbose_name="NumWebVisitsMonth")
    accepted_cmp3 = models.IntegerField(verbose_name="AcceptedCmp3")
    accepted_cmp4 = models.IntegerField(verbose_name="AcceptedCmp4")
    accepted_cmp5 = models.IntegerField(verbose_name="AcceptedCmp5")
    accepted_cmp1 = models.IntegerField(verbose_name="AcceptedCmp1")
    accepted_cmp2 = models.IntegerField(verbose_name="AcceptedCmp2")
    complain = models.IntegerField(verbose_name="Complain")
    age = models.IntegerField(verbose_name="Age")
    education_encoded = models.FloatField(verbose_name="Education_Encoded")
    marital_absurd = models.FloatField(verbose_name="Marital_Absurd")
    marital_alone = models.FloatField(verbose_name="Marital_Alone")
    marital_divorced = models.FloatField(verbose_name="Marital_Divorced")
    marital_married = models.FloatField(verbose_name="Marital_Married")
    marital_single = models.FloatField(verbose_name="Marital_Single")
    marital_together = models.FloatField(verbose_name="Marital_Together")
    marital_widow = models.FloatField(verbose_name="Marital_Widow")
    marital_yolo = models.FloatField(verbose_name="Marital_YOLO")
    
    # Outputs
    prediction_response = models.IntegerField(verbose_name="Response", null=True, blank=True)
    confidence_score = models.FloatField(verbose_name="Confidence Score", null=True, blank=True)

    def __str__(self):
        return f"Prediction {self.id} on {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}"

