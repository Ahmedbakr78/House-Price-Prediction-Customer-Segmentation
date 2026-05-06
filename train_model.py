import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import joblib
import os
from datetime import datetime

# Create models directory
if not os.path.exists('models'):
    os.makedirs('models')

print("🚀 Starting Professional Model Training Pipeline...")

# 1. Load Data
print("\n1️⃣ Loading Data...")
df = pd.read_csv('archive (2)/output.csv')
print(f"   Original shape: {df.shape}")

# 2. Basic Cleaning & Feature Engineering
print("\n2️⃣ Feature Engineering & Cleaning...")

# Drop duplicates
df = df.drop_duplicates()

# Feature Engineering
current_year = datetime.now().year
df['house_age'] = current_year - df['yr_built']
df['years_since_renovation'] = df['yr_renovated'].apply(lambda x: current_year - x if x > 0 else 0)
df['total_sqft'] = df['sqft_living'] + df['sqft_lot']

# Handle Outliers (Log Transformation for Target)
# We will train on log(price) to handle skewness
df['price_log'] = np.log1p(df['price'])

# Drop unnecessary columns
cols_to_drop = ['date', 'street', 'statezip', 'country', 'yr_built', 'yr_renovated'] 
# We keep 'city' for encoding
df_processed = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# Save processed data for EDA (with new features)
df_processed.to_csv('models/processed_data.csv', index=False)
print("   Saved processed_data.csv with engineered features.")

# Prepare X and y
X = df_processed.drop(['price', 'price_log'], axis=1)
y = df_processed['price_log'] # Train on Log Price

# Identify columns
categorical_cols = ['city']
numerical_cols = [col for col in X.columns if col not in categorical_cols]

print(f"   Numerical Features: {len(numerical_cols)}")
print(f"   Categorical Features: {len(categorical_cols)}")

# 3. Split Data
print("\n3️⃣ Splitting Data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Build Pipeline
print("\n4️⃣ Building Advanced Pipeline...")

# Numerical Pipeline: KNN Imputation -> Scaling
numeric_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', StandardScaler())
])

# Categorical Pipeline: OneHot Encoding
categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Model Placeholder (will be replaced by SearchCV)
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', XGBRegressor())])

# 5. Hyperparameter Tuning (XGBoost vs LightGBM)
print("\n5️⃣ Hyperparameter Tuning with RandomizedSearchCV...")

param_distributions = [
    {
        'regressor': [XGBRegressor(random_state=42, n_jobs=-1)],
        'regressor__n_estimators': [100, 300, 500],
        'regressor__learning_rate': [0.01, 0.05, 0.1],
        'regressor__max_depth': [3, 5, 7],
        'regressor__subsample': [0.7, 0.9, 1.0],
        'preprocessor__num__imputer__n_neighbors': [3, 5, 7]
    },
    {
        'regressor': [LGBMRegressor(random_state=42, n_jobs=-1, verbose=-1)],
        'regressor__n_estimators': [100, 300, 500],
        'regressor__learning_rate': [0.01, 0.05, 0.1],
        'regressor__num_leaves': [31, 50, 100],
        'preprocessor__num__imputer__n_neighbors': [3, 5, 7]
    }
]

search = RandomizedSearchCV(
    pipeline, 
    param_distributions, 
    n_iter=10, # Number of parameter settings that are sampled
    cv=3, 
    verbose=1, 
    scoring='neg_mean_squared_error', 
    random_state=42,
    n_jobs=-1
)

search.fit(X_train, y_train)

best_model = search.best_estimator_
print(f"\n✅ Best Model Found: {best_model.named_steps['regressor']}")
print(f"   Best Params: {search.best_params_}")

# 6. Evaluation
print("\n6️⃣ Evaluating Best Model...")

# Predict (Log scale)
y_pred_log = best_model.predict(X_test)

# Inverse Transform (to get actual prices)
y_pred = np.expm1(y_pred_log)
y_test_actual = np.expm1(y_test)

r2 = r2_score(y_test_actual, y_pred)
mae = mean_absolute_error(y_test_actual, y_pred)
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))

print(f"   R² Score: {r2:.4f}")
print(f"   MAE: ${mae:,.2f}")
print(f"   RMSE: ${rmse:,.2f}")

# 7. Save Artifacts
print("\n7️⃣ Saving Artifacts...")
joblib.dump(best_model, 'models/best_pipeline.pkl')
print("   Saved best_pipeline.pkl")

# Save column names for app usage
feature_names = {
    'numerical': numerical_cols,
    'categorical': categorical_cols
}
joblib.dump(feature_names, 'models/feature_names.pkl')
print("   Saved feature_names.pkl")

print("\n✅ Professional Pipeline Completed Successfully!")
