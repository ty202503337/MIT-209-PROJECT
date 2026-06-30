# Customer Personality Analysis: Campaign Response Prediction

## Goal/Objective of the Project
The primary objective of this project is to build a supervised machine learning classification model capable of predicting whether a customer will respond positively to a marketing campaign. By analyzing customer demographics, purchasing behavior, and historical interaction with previous campaigns, the model helps businesses optimize their marketing strategies. Instead of spending resources targeting all customers, the business can accurately identify the segment most likely to accept the offer, thereby increasing conversion rates and maximizing return on investment (ROI).

## Dataset
**Dataset URL:** [Customer Personality Analysis on Kaggle](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis/data)

The dataset contains detailed information regarding the company's customer base, structured into various attributes:

### Target Variable
* **Response:** 1 if the customer accepted the offer in the last campaign, 0 otherwise.

### Features/Columns Description

**People (Demographics & History)**
* **ID:** Customer's unique identifier.
* **Year_Birth:** Customer's birth year.
* **Education:** Customer's highest education level.
* **Marital_Status:** Customer's marital status.
* **Income:** Customer's yearly household income.
* **Kidhome:** Number of children in the customer's household.
* **Teenhome:** Number of teenagers in the customer's household.
* **Dt_Customer:** Date of the customer's enrollment with the company.
* **Recency:** Number of days since the customer's last purchase.
* **Complain:** 1 if the customer complained in the last 2 years, 0 otherwise.

**Products (Spending Habits over the last 2 years)**
* **MntWines:** Amount spent on wine.
* **MntFruits:** Amount spent on fruits.
* **MntMeatProducts:** Amount spent on meat products.
* **MntFishProducts:** Amount spent on fish products.
* **MntSweetProducts:** Amount spent on sweets.
* **MntGoldProds:** Amount spent on gold products.

**Promotion (Campaign History & Deal Usage)**
* **NumDealsPurchases:** Number of purchases made with a discount.
* **AcceptedCmp1:** 1 if the customer accepted the offer in the 1st campaign, 0 otherwise.
* **AcceptedCmp2:** 1 if the customer accepted the offer in the 2nd campaign, 0 otherwise.
* **AcceptedCmp3:** 1 if the customer accepted the offer in the 3rd campaign, 0 otherwise.
* **AcceptedCmp4:** 1 if the customer accepted the offer in the 4th campaign, 0 otherwise.
* **AcceptedCmp5:** 1 if the customer accepted the offer in the 5th campaign, 0 otherwise.

**Place (Channels of Purchase)**
* **NumWebPurchases:** Number of purchases made through the company's website.
* **NumCatalogPurchases:** Number of purchases made using a catalog.
* **NumStorePurchases:** Number of purchases made directly in stores.
* **NumWebVisitsMonth:** Number of visits to the company's website in the last month.

*(Note: Features like `Z_CostContact` and `Z_Revenue` have constant values across the dataset and are typically removed during preprocessing).*

## Data Cleaning and Preparation
To ensure the model receives high-quality data, the following techniques were utilized for data preprocessing:
* **Handling Missing Values:** Missing values (primarily found in the `Income` column) were imputed using the median to avoid skewing the distribution with extreme values.
* **Date/Time Formatting:** The `Dt_Customer` feature was converted to a standard `datetime` format.
* **Feature Engineering:** * Derived an `Age` feature by subtracting `Year_Birth` from the current data year.
    * Created an `Enrollment_Duration` (or Tenure) feature to measure customer loyalty based on `Dt_Customer`.
    * Aggregated spending columns to create a `Total_Spend` feature.
    * Combined `Kidhome` and `Teenhome` to create a `Total_Dependents` feature.
* **Dropping Irrelevant Features:** Removed identifier columns like `ID` and constant variables like `Z_CostContact` and `Z_Revenue` as they hold no predictive power.
* **Categorical Encoding:** Applied One-Hot Encoding and Label Encoding to convert categorical text data (e.g., `Education`, `Marital_Status`) into numeric formats. `Marital_Status` anomalies (like 'YOLO' or 'Absurd') were merged into broader logical categories (e.g., 'Single').
* **Feature Scaling:** Standardized numerical variables using `StandardScaler` to bring all features to a similar scale, optimizing algorithm convergence.
* **Class Imbalance Treatment:** Applied techniques like SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalances in the target `Response` variable, preventing the model from becoming biased toward the majority class.

## Algorithms and Model Evaluation

### Algorithms Used
Various classification algorithms were trained and tested to find the best performing model for predicting campaign response:
* **Logistic Regression:** Serving as a strong, interpretable baseline model.
* **Decision Tree Classifier:** To capture non-linear relationships and segment boundaries.
* **Random Forest Classifier:** An ensemble method used to reduce overfitting and improve predictive accuracy.
* **Gradient Boosting Classifiers (XGBoost / LightGBM):** Advanced ensemble techniques that sequentially correct errors from previous decision trees, highly effective on tabular data.
* **Support Vector Machines (SVM):** Used to find the optimal hyper-plane separating customers who accept offers from those who do not.

### Evaluation Metrics
To robustly assess model performance beyond simple accuracy, the following metrics were evaluated:
* **Accuracy:** The overall percentage of correctly predicted responses.
* **Precision:** The proportion of predicted positive responses that were actually positive (minimizing false positives to save marketing costs).
* **Recall (Sensitivity):** The proportion of actual positive responses correctly identified by the model (minimizing false negatives to avoid missing out on potential buyers).
* **F1-Score:** The harmonic mean of Precision and Recall, providing a balanced metric—especially useful due to target class imbalances.
* **ROC-AUC (Receiver Operating Characteristic - Area Under Curve):** Measures the model's ability to distinguish between the two classes (responders vs. non-responders) across various threshold levels. 
* **Confusion Matrix:** Utilized to visualize the breakdown of True Positives, True Negatives, False Positives, and False Negatives.