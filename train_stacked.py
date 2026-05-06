import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
import os
import time
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

if not os.path.exists('models'):
    os.makedirs('models')

print("="*50)
print("  STACKED ENSEMBLE TRAINING (SUPER MODEL)")
print("  By Ahmed Abobakr")
print("="*50)

# --- 1. Load & Clean Data ---
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Loading & Cleaning...")
df = pd.read_csv('archive (2)/output.csv')

# Basic Cleaning
df = df.drop_duplicates()
df = df[df['price'] > 0]

# Remove Outliers (IQR)
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]

# --- 2. Feature Engineering ---
current_year = datetime.now().year
df['house_age'] = current_year - df['yr_built']
df['years_since_renovation'] = df['yr_renovated'].apply(lambda x: current_year - x if x > 0 else 0)
df['total_sqft'] = df['sqft_living'] + df['sqft_lot']
df['price_log'] = np.log1p(df['price'])

# Select Features
features_to_drop = ['date', 'street', 'statezip', 'country', 'yr_built', 'yr_renovated', 'price']
X = df.drop(columns=[c for c in features_to_drop if c in df.columns])
if 'price_log' in X.columns: X = X.drop(columns=['price_log'])
y = df['price_log']

categorical_cols = ['city']
numerical_cols = [col for col in X.columns if col not in categorical_cols]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 3. Build Preprocessor ---
numeric_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', RobustScaler())
])
categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# --- 4. Define Base Models for Stacking ---
estimators = [
    ('xgb', XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)),
    ('rf', RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)),
    ('gb', GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42))
]

# --- 5. Create Stacking Regressor ---
# The 'Final Estimator' learns how to best combine the predictions of the base models
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Training Stacked Ensemble (This may take a moment)...")

stacked_model = StackingRegressor(
    estimators=estimators,
    final_estimator=RidgeCV(), # Simple linear model to combine predictions
    n_jobs=-1,
    cv=3 # 3-fold cross-validation
)

pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', stacked_model)])

start_time = time.time()
pipeline.fit(X_train, y_train)
training_time = time.time() - start_time

# --- 6. Evaluate ---
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Evaluating...")
y_pred_log = pipeline.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_test_actual = np.expm1(y_test)

r2 = r2_score(y_test_actual, y_pred)
mae = mean_absolute_error(y_test_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))

print(f"\n{'-'*30}")
print(f"STACKED MODEL ACCURACY (R2): {r2:.2%}")
print(f"{'-'*30}")

# --- 7. Save & Update Results ---
# We do NOT overwrite the default 'best_pipeline.pkl' yet, unless you want the app to use this by default.
# Let's save it as 'stacked_pipeline.pkl' specifically for the new Tab.
joblib.dump(pipeline, 'models/stacked_pipeline.pkl')

# Update CSV
try:
    existing_df = pd.read_csv('models/final_comparison.csv')
    # Filter out old stacked runs if any
    existing_df = existing_df[existing_df['Model'] != 'Stacked_Ensemble']
    
    new_row = pd.DataFrame([{
        'Model': 'Stacked_Ensemble',
        'R2_Score': r2,
        'MAE': mae,
        'RMSE': rmse,
        'Training_Time_s': training_time,
        'Training_Mode': 'Ensemble (Super Model)'
    }])
    
    final_df = pd.concat([existing_df, new_row], ignore_index=True).sort_values('R2_Score', ascending=False)
    final_df.to_csv('models/final_comparison.csv', index=False)
    # Update main comparison too
    final_df.to_csv('models/model_comparison.csv', index=False)
    print("Results updated.")
except Exception as e:
    print(f"Could not update CSV: {e}")

print("\nDone! Stacked model saved.")
