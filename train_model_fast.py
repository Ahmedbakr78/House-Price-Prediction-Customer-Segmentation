import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
import os
from datetime import datetime
import time

if not os.path.exists('models'):
    os.makedirs('models')

print("starting professional model training pipeline...")

print("\nloading data...")
df = pd.read_csv('archive (2)/output.csv')
print(f"original shape: {df.shape}")

print("\nfeature engineering and cleaning...")
df = df.drop_duplicates()

current_year = datetime.now().year
df['house_age'] = current_year - df['yr_built']
df['years_since_renovation'] = df['yr_renovated'].apply(lambda x: current_year - x if x > 0 else 0)
df['total_sqft'] = df['sqft_living'] + df['sqft_lot']

df['price_log'] = np.log1p(df['price'])

cols_to_drop = ['date', 'street', 'statezip', 'country', 'yr_built', 'yr_renovated']
df_processed = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

df_processed.to_csv('models/processed_data.csv', index=False)
print("saved processed data")

X = df_processed.drop(['price', 'price_log'], axis=1)
y = df_processed['price_log']

categorical_cols = ['city']
numerical_cols = [col for col in X.columns if col not in categorical_cols]

print(f"numerical features: {len(numerical_cols)}")
print(f"categorical features: {len(categorical_cols)}")

print("\nsplitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nbuilding pipeline...")
numeric_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

print("\ntraining multiple models...")
models = {
    'gradient_boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
    'xgboost': XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1),
    'lightgbm': LGBMRegressor(n_estimators=100, num_leaves=31, learning_rate=0.1, random_state=42, n_jobs=-1, verbose=-1)
}

results = {}

for name, model in models.items():
    print(f"\ntraining {name}...")
    start_time = time.time()
    
    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
    pipeline.fit(X_train, y_train)
    
    y_pred_log = pipeline.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test_actual = np.expm1(y_test)
    
    r2 = r2_score(y_test_actual, y_pred)
    mae = mean_absolute_error(y_test_actual, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
    training_time = time.time() - start_time
    
    results[name] = {
        'pipeline': pipeline,
        'r2': r2,
        'mae': mae,
        'rmse': rmse,
        'training_time': training_time
    }
    
    print(f"r2 score: {r2:.4f}")
    print(f"mae: ${mae:,.2f}")
    print(f"rmse: ${rmse:,.2f}")
    print(f"training time: {training_time:.2f}s")

print("\nselecting best model...")
best_model_name = max(results, key=lambda x: results[x]['r2'])
best_pipeline = results[best_model_name]['pipeline']

print(f"best model: {best_model_name}")
print(f"r2 score: {results[best_model_name]['r2']:.4f}")

print("\nsaving artifacts...")
joblib.dump(best_pipeline, 'models/best_pipeline.pkl')
joblib.dump(results, 'models/model_comparison.pkl')

feature_names = {
    'numerical': numerical_cols,
    'categorical': categorical_cols
}
joblib.dump(feature_names, 'models/feature_names.pkl')

comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'R2_Score': [results[k]['r2'] for k in results.keys()],
    'MAE': [results[k]['mae'] for k in results.keys()],
    'RMSE': [results[k]['rmse'] for k in results.keys()],
    'Training_Time_s': [results[k]['training_time'] for k in results.keys()]
})
comparison_df = comparison_df.sort_values('R2_Score', ascending=False)
comparison_df.to_csv('models/model_comparison.csv', index=False)

print("\npipeline completed successfully")
print("\nmodel comparison summary:")
print(comparison_df.to_string(index=False))
