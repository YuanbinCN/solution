import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score, classification_report
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
import smogn
import os

targets = {
    "CL": "Sale_CL",    
    "CC": "Sale_CC",    
    "MF": "Sale_MF" 
}

feature_columns = {
    "transaction": [
        "TransactionsCred", "TransactionsDeb", "TransactionsDebCashless_Card",
        "TransactionsDebCash_Card", "TransactionsDeb_PaymentOrder"
    ],
    "numeric": [
        "Count_CA", "Count_SA", "Count_MF", "Count_CL", "Count_CC",
        "ActBal_CA", "ActBal_SA", "ActBal_MF", "ActBal_CL", 
        "ActBal_CC", "ActBal_OVD",
        "Age", "Tenure"
    ],
    "categorical": [
        "Sex"
    ]
}


class ProductPropensityModel:
    def __init__(self, target_product):
        self.target_product = target_product
        
        self.transaction_features = feature_columns["transaction"]
        self.other_numeric_features = feature_columns["numeric"]
        self.categorical_features = feature_columns["categorical"]
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('transaction_num', Pipeline([
                    ('imputer', KNNImputer(n_neighbors=5)),
                    ('scaler', StandardScaler())
                ]), self.transaction_features),
                ('other_num', Pipeline([
                    ('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                    ('scaler', StandardScaler())
                ]), self.other_numeric_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), self.categorical_features)
            ])
        
        base_classifier = xgb.XGBClassifier(
            objective='binary:logistic',
            random_state=42,
            learning_rate=0.01,        
            max_depth=10,             
            n_estimators=100,         
        )
        
        self.model = ImbPipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('rfe', RFE(estimator=base_classifier, n_features_to_select=10)), 
            ('smote', SMOTE(random_state=42)),
            ('classifier', base_classifier)
        ])

    def train(self, data, n_splits=5, random_state=42):
        target_col = targets[self.target_product]
        X = data[self.transaction_features + self.other_numeric_features + self.categorical_features]
        y = data[target_col]
        
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        self.cv_scores = cross_val_score(
            self.model, X, y, 
            cv=cv, 
            scoring='roc_auc',
            n_jobs=-1 
        )
        
        self.model.fit(X, y)
    
    def evaluate(self):
        print(f"\n{self.target_product} Model Metrics:")
        print(f"Mean ROC AUC: {self.cv_scores.mean():.3f} (+/- {self.cv_scores.std() * 2:.3f})")
    
    def predict_proba(self, X):  
        return self.model.predict_proba(X)[:, 1]

class RevenueModel(ProductPropensityModel):
    def __init__(self, target_product):
        super().__init__(target_product)
        
        self.model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', xgb.XGBRegressor(
                objective='reg:squarederror',
                random_state=42,
                learning_rate=0.01,
                max_depth=10,
                n_estimators=100,
            ))
        ])

    def train(self, data, n_splits=5, random_state=42):
        target_col = f"Revenue_{self.target_product}"
        X = data[self.transaction_features + self.other_numeric_features + self.categorical_features]
        y = data[target_col]

        cv = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        self.cv_scores = -cross_val_score(
            self.model, X, y, 
            cv=cv,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1
        )
        
        self.model.fit(X, y)

    def evaluate(self):
        print(f"\n{self.target_product} Revenue Model Metrics:")
        print(f"Mean RMSE: {abs(self.cv_scores.mean()):.2f} (+/- {self.cv_scores.std() * 2:.2f})")

    def predict(self, X):  
        return self.model.predict(X)

def analyze_customer_opportunities(training_data, test_data):
    propensity_models = {}
    for product in ["CC", "MF", "CL"]:
        model = ProductPropensityModel(product)
        model.train(training_data)
        model.evaluate()
        propensity_models[product] = model
    
    revenue_models = {}
    for product in ["CC", "MF", "CL"]:
        model = RevenueModel(product)
        model.train(training_data)
        model.evaluate()
        revenue_models[product] = model
    
    # Calculate probabilities and revenues for each product using their specific features
    probabilities = {}
    revenues = {}
    for product, model in propensity_models.items():
        test_features = test_data[model.transaction_features + 
                                model.other_numeric_features + 
                                model.categorical_features]
        probabilities[product] = model.predict_proba(test_features)
        revenues[product] = revenue_models[product].predict(test_features)
    
    expected_revenues = {
        product: probabilities[product] * revenues[product]
        for product in ["CC", "MF", "CL"]
    }
    
    results = pd.DataFrame({
        'Customer_ID': test_features.iloc[:, 0],
        'CC_Revenue': expected_revenues["CC"],
        'MF_Revenue': expected_revenues["MF"],
        'CL_Revenue': expected_revenues["CL"]
    })
    
    results['CC_Probability'] = probabilities["CC"]
    results['MF_Probability'] = probabilities["MF"]
    results['CL_Probability'] = probabilities["CL"]
    
    results['Best_Product'] = results[['CC_Revenue', 'MF_Revenue', 'CL_Revenue']].idxmax(axis=1).map({
        'CC_Revenue': 'CC',
        'MF_Revenue': 'MF', 
        'CL_Revenue': 'CL'
    })
    results['Best_Revenue'] = results[['CC_Revenue', 'MF_Revenue', 'CL_Revenue']].max(axis=1)
    
    print("\nTop 100 revenue recommendations:")
    top_100 = results.nlargest(100, 'Best_Revenue')
    print(top_100[['Customer_ID', 'Best_Product', 'Best_Revenue']])
    # Save results to Excel file
    output_path = "./results/recommendations.xlsx"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    top_100.to_excel(output_path, index=False)
    print(f"\nResults saved to {output_path}")
    return results, probabilities

if __name__ == "__main__":
    data = pd.read_excel("./dataset/training.xlsx")
    test_data = pd.read_excel("./dataset/testing.xlsx")
    results, probabilities = analyze_customer_opportunities(data, test_data)
