import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
import os
from datetime import datetime
import time
import warnings

warnings.filterwarnings('ignore')

if not os.path.exists('models'):
    os.makedirs('models')

print("="*50)
print("  HIGH ACCURACY TRAINING PIPELINE (5*A)")
print("  By Ahmed Abobakr")
print("="*50)

print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Loading data...")
df = pd.read_csv('archive (2)/output.csv')
print(f"Original shape: {df.shape}")

# --- 1. Advanced Preprocessing & Cleaning ---
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Cleaning data & Removing Outliers...")

# Remove duplicates
df = df.drop_duplicates()

# Handle logical errors (e.g. 0 price)
df = df[df['price'] > 0]

# --- Outlier Removal (Crucial for Regression) ---
# Using IQR method for Price
Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

old_len = len(df)
df = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
print(f"Removed {old_len - len(df)} outliers based on Price.")

# --- 2. Feature Engineering ---
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Engineering Features...")
current_year = datetime.now().year

# Age of house
df['house_age'] = current_year - df['yr_built']

# Years since renovation (0 if never)
df['years_since_renovation'] = df['yr_renovated'].apply(lambda x: current_year - x if x > 0 else 0)

# Total square footage
df['total_sqft'] = df['sqft_living'] + df['sqft_lot']

# Log transform target variable (Normalizes distribution)
df['price_log'] = np.log1p(df['price'])

# Select features
features_to_drop = ['date', 'street', 'statezip', 'country', 'yr_built', 'yr_renovated', 'price']
X = df.drop(columns=[c for c in features_to_drop if c in df.columns])
# Remove price_log from X if it accidentally got in (it shouldn't, but safety first)
if 'price_log' in X.columns:
    X = X.drop(columns=['price_log'])
    
y = df['price_log']

# Define Column types
categorical_cols = ['city'] # We keep City
numerical_cols = [col for col in X.columns if col not in categorical_cols]

print(f"Numerical Features: {len(numerical_cols)} {numerical_cols}")
print(f"Categorical Features: {len(categorical_cols)} {categorical_cols}")

# --- 3. Save Processed Data for App ---
# We need to save a version with 'price_log' for the app visualizations
df_export = X.copy()
df_export['price_log'] = y
df_export.to_csv('models/processed_data.csv', index=False)
print("Saved processed_data.csv")

# --- 4. Split Data ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 5. Build Advanced Pipeline ---
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Building Pipeline...")

# Numeric: Impute missing -> RobustScaler (better for outliers)
numeric_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', RobustScaler()) 
])

# Categorical: OneHot
categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# --- 6. Model Training with Optimized Hyperparameters ---
# We will use XGBoost as it's generally best, but tuned.
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Training Optimized XGBoost Model...")

# Optimized parameters for better accuracy (slower but worth it)
model = XGBRegressor(
    n_estimators=1000,       # More trees
    learning_rate=0.05,      # Slower learning for better convergence
    max_depth=6,             # Deeper trees
    min_child_weight=1,
    subsample=0.8,           # Reduce overfitting
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42
)

pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])

start_time = time.time()
pipeline.fit(X_train, y_train)
training_time = time.time() - start_time

# --- 7. Evaluation ---
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Evaluating...")

y_pred_log = pipeline.predict(X_test)
y_pred = np.expm1(y_pred_log)       # Inverse log
y_test_actual = np.expm1(y_test)    # Inverse log

r2 = r2_score(y_test_actual, y_pred)
mae = mean_absolute_error(y_test_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))

print(f"\n{'-'*30}")
print(f"FINAL ACCURACY (R2 Score): {r2:.2%}")
print(f"{'-'*30}")
print(f"MAE:  ${mae:,.2f}")
print(f"RMSE: ${rmse:,.2f}")
print(f"Training Time: {training_time:.2f}s")

# --- 8. Save Artifacts ---
print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Saving Artifacts...")

# Save the pipeline (This overwrites the 'best_pipeline.pkl' so the App uses THIS model)
joblib.dump(pipeline, 'models/best_pipeline.pkl')

feature_names = {
    'numerical': numerical_cols,
    'categorical': categorical_cols
}
joblib.dump(feature_names, 'models/feature_names.pkl')

# Update comparison CSV just to record this run
results = {
    'Optimized_XGBoost': {
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'training_time': training_time
    }
}
# We'll just create a simple DF for the record
comparison_df = pd.DataFrame({
    'Model': ['Optimized_XGBoost'],
    'R2_Score': [r2],
    'MAE': [mae],
    'RMSE': [rmse],
    'Training_Time_s': [training_time]
})
comparison_df.to_csv('models/model_comparison.csv', index=False)

print("\nSUCCESS! New high-accuracy model saved.")
print(f"You can now run 'streamlit run app.py' to see the improved predictions.")
