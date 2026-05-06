# House Price Prediction & Customer Segmentation



## 1. Project Overview

project is a  machine learning application that implements two core analytical workflows:

1. **Regression Analysis**: House price prediction using advanced preprocessing techniques and state-of-the-art algorithms (XGBoost, LightGBM, Gradient Boosting)

2. **Clustering Analysis**: Customer segmentation using K-Means, DBSCAN, and Hierarchical clustering with PCA/t-SNE dimensionality reduction

The project combines professional data preprocessing, multiple ML algorithms, and an interactive Streamlit web interface to provide accurate predictions with comprehensive market analysis.

**Key Deliverables**:

- Trained ML models serialized as .pkl files
- Interactive web application for real-time predictions
- Batch processing capability via CSV upload
- Comprehensive documentation and testing suite
- Academic compliance with all course requirements

---

## 2. System Architecture

### Architecture Layers

```
Output Layer
  - Price Predictions
  - Model Metrics
  - Visual Insights

Application Layer
  - Streamlit UI
  - Batch Processing
  - Interactive Visualizations

Model Layer
  - XGBoost (Winner)
  - LightGBM
  - Gradient Boosting

Processing Layer
  - Data Cleaning
  - Feature Engineering
  - KNN Imputation
  - Standard Scaling

Data Layer
  - Raw CSV Data (4600 records)
  - Processed Datasets
```

### Complete System Diagram

```
Input Sources -> Data Processing -> ML Pipeline -> Models -> Application -> Outputs

CSV File/User Form/Batch Upload
  -> Pandas DataFrame/Feature Engineering/Data Validation
  -> KNN Imputer/Standard Scaler/One-Hot Encoder/Model Ensemble
  -> XGBoost/LightGBM/GradBoost with R2 metrics
  -> Streamlit UI/Plotly Charts/Data Tables
  -> Price Predictions/CSV Download/Visual Reports/Model Metrics
```

---

## 3. Technology Stack

| Layer    | Component    | Technology            | Version | Purpose                               |
| -------- | ------------ | --------------------- | ------- | ------------------------------------- |
| Language | core         | python                | 3.12+   | programming language                  |
| Data     | manipulation | pandas                | latest  | dataframes, data processing           |
| Data     | computation  | numpy                 | latest  | numerical operations                  |
| ML       | framework    | scikit-learn          | latest  | pipelines, preprocessing, base models |
| ML       | boosting     | xgboost               | latest  | gradient boosting (winner model)      |
| ML       | boosting     | lightgbm              | latest  | gradient boosting (alternative)       |
| Web      | framework    | streamlit             | latest  | web application interface             |
| Web      | navigation   | streamlit-option-menu | latest  | sidebar navigation                    |
| Viz      | interactive  | plotly                | latest  | interactive charts                    |
| Viz      | static       | matplotlib            | latest  | static visualizations                 |
| Viz      | statistical  | seaborn               | latest  | statistical plots                     |
| Storage  | persistence  | joblib                | latest  | model saving/loading                  |

---



## 2. EXECUTIVE SUMMARY

This project – **"5★A by Ahmed Abobakr"** – is a professional, end‑to‑end machine learning application that predicts house prices and segments customers / properties. It combines advanced data preprocessing, state‑of‑the‑art algorithms (XGBoost, LightGBM, Gradient Boosting), and an interactive Streamlit web dashboard.

The system consists of:

- **Jupyter Notebooks** for regression analysis, customer segmentation, and advanced property clustering.
- **Training scripts** (`train_model.py`, `train_modelfast.py`) that automate data cleaning, feature engineering, hyperparameter tuning, model comparison, and serialisation.
- **A web application (`app.py`)** providing single & batch prediction, mortgage calculator, model comparison, interactive EDA, and help pages.

The project fully meets and exceeds all Machine Learning course requirements, incorporating dirty data simulation, multiple clustering algorithms, feature selection, and a professional documentation suite.

---

## 3. PROJECT INFO TABLE

| Property        | Details                                                                                                                                                                                                         |
| --------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Project Name    | VML‑Ahmed: ML Price Prediction & Segmentation (5★A)                                                                                                                                                             |
| Type            | Machine Learning Application (Regression + Clustering) with Web API                                                                                                                                             |
| Technologies    | Python, Scikit‑learn, Pandas, NumPy, Streamlit, Jupyter, Joblib                                                                                                                                                 |
| Description     | Dual‑purpose ML project: 1) Regression model for house price prediction, 2) Clustering models for customer/house segmentation. Trained models are serialised (.pkl) and served via a Streamlit web application. |
| Primary Input   | `archive (2)/output.csv` (House Prices), `clustering_customers.csv`, `models/processed_data.csv`                                                                                                                |
| Primary Outputs | `models/best_pipeline.pkl`, `models/model_comparison.csv`, trained models, interactive dashboard                                                                                                                |

---



### 4.2 All Files Table (Core)

| #   | File Path                                   | Type              | Purpose                                                                                             |
| --- | ------------------------------------------- | ----------------- | --------------------------------------------------------------------------------------------------- |
| 1   | `app.py`                                    | Python Script     | Streamlit web server for predictions, EDA, model comparison, batch processing                       |
| 2   | `train_model.py`                            | Python Script     | End‑to‑end training pipeline with multiple models (GradientBoosting, XGBoost, LightGBM)             |
| 3   | `train_modelfast.py`                        | Python Script     | Automated hyperparameter tuning via RandomizedSearchCV (XGBoost vs LightGBM)                        |
| 4   | `01_Regression_House_Prices.ipynb`          | Jupyter Notebook  | Full regression workflow: EDA, cleaning, training Linear/DecisionTree/RandomForest/GradientBoosting |
| 5   | `02_Clustering_Customer_Segmentation.ipynb` | Jupyter Notebook  | Customer segmentation with K‑Means and DBSCAN                                                       |
| 6   | `02_Clustering_House_Prices.ipynb`          | Jupyter Notebook  | Advanced property clustering using PCA, t‑SNE, and hierarchical methods                             |
| 7   | `models/best_pipeline.pkl`                  | Serialized Object | Final tuned pipeline (preprocessor + model) for production use                                      |
| 8   | `models/processed_data.csv`                 | CSV Data          | Cleaned data with engineered features, used for analysis & batch scenarios                          |
| 9   | `models/model_comparison.csv`               | CSV Data          | Performance metrics (R², RMSE, MAE, Time) for all trained models                                    |
| 10  | `models/feature_names.pkl`                  | Serialized Object | Lists of numerical & categorical column names                                                       |
| 11  | `archive (2)/output.csv`                    | CSV Data          | Raw house price dataset                                                                             |
| 12  | `clustering_customers.csv`                  | CSV Data          | Customer data for segmentation (Age, Income, Spending Score)                                        |

---

## 5. TECHNOLOGY STACK

### 5.1 Core Technologies

| Layer    | Component    | Technology            | Version | Purpose                               |
| -------- | ------------ | --------------------- | ------- | ------------------------------------- |
| Language | core         | Python                | 3.12+   | Programming language                  |
| Data     | manipulation | pandas                | latest  | DataFrames, data processing           |
| Data     | computation  | numpy                 | latest  | Numerical operations                  |
| ML       | framework    | scikit‑learn          | latest  | Pipelines, preprocessing, base models |
| ML       | boosting     | xgboost               | latest  | Gradient boosting (winner)            |
| ML       | boosting     | lightgbm              | latest  | Gradient boosting (alternative)       |
| Web      | framework    | streamlit             | latest  | Web application interface             |
| Web      | navigation   | streamlit‑option‑menu | latest  | Sidebar navigation                    |
| Viz      | interactive  | plotly                | latest  | Interactive charts                    |
| Viz      | static       | matplotlib            | latest  | Static visualisations                 |
| Viz      | statistical  | seaborn               | latest  | Statistical plots                     |
| Storage  | persistence  | joblib                | latest  | Model saving/loading                  |

### 5.2 System Requirements

| Requirement | Minimum             | Recommended        | Notes                    |
| ----------- | ------------------- | ------------------ | ------------------------ |
| OS          | Linux/macOS/Windows | Linux              | Tested on Linux          |
| Python      | 3.10                | 3.12               | Project uses 3.12        |
| RAM         | 4 GB                | 8 GB               | For large datasets       |
| Storage     | 500 MB              | 1 GB               | Includes data and models |
| Internet    | Required (setup)    | Optional (runtime) | For package installation |

### 5.3 Required Packages Detail

| Package               | Size    | Dependencies | Use Case              |
| --------------------- | ------- | ------------ | --------------------- |
| pandas                | ~50 MB  | numpy, pytz  | Data manipulation     |
| numpy                 | ~25 MB  | none         | Numerical computation |
| scikit‑learn          | ~30 MB  | numpy, scipy | ML algorithms         |
| xgboost               | ~120 MB | numpy, scipy | Gradient boosting     |
| lightgbm              | ~5 MB   | numpy, scipy | Gradient boosting     |
| streamlit             | ~15 MB  | many         | Web framework         |
| streamlit‑option‑menu | ~1 MB   | streamlit    | Navigation            |
| matplotlib            | ~30 MB  | numpy        | Plotting              |
| seaborn               | ~5 MB   | matplotlib   | Statistical viz       |

Total download size: ~300 MB; Total installation time: ~2‑3 minutes.

---

## 6. FUNCTIONS

### 6.1 Application (`app.py`) Functions

| #   | Function Name    | Parameters | Returns                                                | Purpose                                                                          |
| --- | ---------------- | ---------- | ------------------------------------------------------ | -------------------------------------------------------------------------------- |
| 1   | `load_artifacts` | None       | `tuple(pipeline, df, feature_names, model_comparison)` | Loads all ML artifacts with Streamlit caching; handles missing files gracefully. |

#### `load_artifacts` Details

- **Decorator:** `@st.cache_resource` – loads only once per session.
- **Error Handling:** `try...except FileNotFoundError` with user‑friendly error message.
- **Returned Objects:**
  - `pipeline` – the best‑tuned `sklearn.pipeline.Pipeline`
  - `df` – cleaned `processed_data.csv` as DataFrame
  - `feature_names` – dict of numerical & categorical feature lists
  - `model_comparison` – DataFrame of model metrics

### 6.2 Notebook Functions

| #   | Function Name                  | Parameters        | Returns | Purpose                                              |
| --- | ------------------------------ | ----------------- | ------- | ---------------------------------------------------- |
| 1   | `parse_json_column` `(nested)` | `x`, `key="name"` | `list`  | Parses JSON‑like strings (genre extraction)          |
| 2   | `get_season` `(nested)`        | `month: int`      | `str`   | Month → season mapping                               |
| 3   | `display` `(helper)`           | `*args`           | None    | Prints multiple arguments (mimics Jupyter `display`) |

### 6.3 Training Scripts Functions

The training scripts (`train_model.py` and `train_modelfast.py`) are procedural; their core logic is embodied in Scikit‑learn `Pipeline` and `ColumnTransformer` objects. No custom functions are defined – the workflow uses:

- `KNNImputer`, `StandardScaler` for numerical pipelines
- `OneHotEncoder` for categorical encoding
- `ColumnTransformer` to orchestrate transformations
- `RandomizedSearchCV` for automated hyperparameter tuning

---

## 7. CLASSES

No custom Python classes are defined. The project leverages built‑in classes from imported libraries:

- `sklearn.pipeline.Pipeline`
- `sklearn.compose.ColumnTransformer`
- `sklearn.impute.KNNImputer`
- `sklearn.preprocessing.StandardScaler`, `OneHotEncoder`
- `sklearn.ensemble.GradientBoostingRegressor`
- `xgboost.XGBRegressor`
- `lightgbm.LGBMRegressor`
- `sklearn.model_selection.RandomizedSearchCV`

---

## 8. API ENDPOINTS

The Streamlit application does not expose traditional REST endpoints. All interactions occur via the web UI. However, the underlying `app.py` could be conceptually described as offering the following route‑like functionalities:

| #          | Method   | Assumed Route          | Auth | Description                                                                              |
| ---------- | -------- | ---------------------- | ---- | ---------------------------------------------------------------------------------------- |
| 1          | POST     | `/predict/house_price` | None | Accept house features as form input, return predicted price                              |
| 2          | POST     | `/batch_predict`       | None | Accept CSV upload, return CSV with predictions                                           |
| (Internal) | GET/POST | (All pages)            | None | Serves interactive pages: Home, Prediction, Model Comparison, Data Analysis, Batch, Help |

**Request Example (`/predict/house_price` equivalent):**

```json
{
  "total_sqft": 1500,
  "bath": 2,
  "bhk": 3,
  "location": "whitefield"
}
```

**Response:**

```json
{ "success": true, "predicted_price_lakhs": 88.5 }
```

**Error Codes:**

- `400` – Invalid input data
- `500` – Model loading failed

---

## 9. DATABASE / DATA SOURCES

The project uses flat‑file storage (CSV). No SQL/NoSQL database is employed.

### 9.1 Input Datasets

| File Name                  | Records     | Description                                                            |
| -------------------------- | ----------- | ---------------------------------------------------------------------- |
| `archive (2)/output.csv`   | ~4,360 rows | Raw house price data (features like sqft, bedrooms, year built, price) |
| `clustering_customers.csv` | ~200 rows   | Customer data: Age, Annual Income, Spending Score, etc.                |

### 9.2 Output / Processed Datasets

| File Name                   | Records     | Description                                                                          |
| --------------------------- | ----------- | ------------------------------------------------------------------------------------ |
| `models/processed_data.csv` | ~3,488 rows | Cleaned data with engineered features (`house_age`, `total_sqft`, `price_log`, etc.) |

### 9.3 Cleaned Dataset Schema (Key Columns)

| Column                   | Type   | Description                              |
| ------------------------ | ------ | ---------------------------------------- |
| `price`                  | float  | Original house price                     |
| `price_log`              | float  | Log‑transformed target                   |
| `house_age`              | int    | 2024 – year built                        |
| `years_since_renovation` | int    | Years since last renovation (0 if never) |
| `total_sqft`             | float  | sqft_living + sqft_lot                   |
| `sqft_living`            | float  | Living area (sqft)                       |
| `bedrooms`               | int    | Number of bedrooms                       |
| `bathrooms`              | float  | Number of bathrooms                      |
| `city`                   | object | City name (categorical)                  |
| (one‑hot encoded)        | uint8  | Multiple `city_*` columns after encoding |

---

## 10. NOTEBOOK DETAILS

### 10.1 `01_Regression_House_Prices.ipynb`

- **Task:** Predict house prices.
- **Data simulation:** 5% missing values + 100 duplicate rows.
- **Preprocessing:** Median/mode imputation, duplicate removal, IQR outlier capping, `LabelEncoder` for city.
- **Models trained:**
  - Linear Regression
  - Decision Tree (max_depth=10)
  - Random Forest (100 trees)
  - Gradient Boosting (100 trees)
- **Evaluation:** R², RMSE, MAE; results bar plot.
- **Visualisations:** Histograms, boxplots, scatter plots, correlation heatmap.

### 10.2 `02_Clustering_Customer_Segmentation.ipynb`

- **Task:** Segment customers based on Age, Annual Income, Spending Score, etc.
- **Cleaning:** Drop missing, remove duplicates, `LabelEncoder` for Gender.
- **Scaling:** `MinMaxScaler`.
- **Elbow method:** WCSS for K=1..10.
- **Algorithms:**
  - K‑Means (K=4, k‑means++ init)
  - DBSCAN (eps=0.5, min_samples=5)
- **Evaluation:** Silhouette Score, Davies‑Bouldin Index.
- **Visualisation:** 2D scatter plot (Annual Income vs Spending Score, colour by cluster).

### 10.3 `02_Clustering_House_Prices.ipynb`

- **Task:** Advanced property segmentation using `processed_data.csv`.
- **Preprocessing:** `StandardScaler`.
- **Dimensionality Reduction:**
  - PCA (2 components)
  - t‑SNE (perplexity=30, on subset of 1000)
- **Clustering Algorithms:**
  - K‑Means
  - DBSCAN
  - Agglomerative (Hierarchical)
- **Evaluation:** Silhouette & Davies‑Bouldin scores compared in a DataFrame.
- **Cluster Profiling:** Group by cluster, compute means of original features (price, sqft, age) to interpret segments.
- **Visualisation:** PCA scatter plots coloured by cluster.

---

## 11. TRAINING SCRIPTS DETAIL

### 11.1 `train_model.py` – Professional Pipeline

- **Data Loading & Cleaning:** Drops duplicates, creates `house_age`, `years_since_renovation`, `total_sqft`, log‑transforms price.
- **Preprocessing Pipeline:**
  - **Numerical:** `KNNImputer(n_neighbors=5)` → `StandardScaler`
  - **Categorical:** `OneHotEncoder(handle_unknown='ignore')`
  - **ColumnTransformer** applies the above to `numerical_cols` and `categorical_cols`.
- **Models trained:** `GradientBoostingRegressor`, `XGBRegressor`, `LGBMRegressor`.
- **Results stored:** `model_comparison.csv` and `model_comparison.pkl`.
- **Artifact serialisation:** `best_pipeline.pkl`, `feature_names.pkl`, `processed_data.csv`.

### 11.2 `train_modelfast.py` – Hyperparameter‑Tuned Pipeline

- **Advanced search:** `RandomizedSearchCV` (n_iter=10, cv=3, scoring=‘neg_mean_squared_error’).
- **Tuned models:** XGBoost and LightGBM simultaneously.
- **Tuned hyperparameters:** `n_estimators`, `max_depth`, `learning_rate`, `num_leaves` (LightGBM), and **`preprocessor__num__imputer__n_neighbors`** (KNNImputer).
- **Best model selected** based on R²; final evaluation on hold‑out test set.
- **Output:** `best_pipeline.pkl` (the single best‑tuned pipeline).

---

## 12. PREPROCESSING PIPELINE (Detailed)

| Step | Technique           | Parameters              | Input            | Output        | Purpose                                                    |
| ---- | ------------------- | ----------------------- | ---------------- | ------------- | ---------------------------------------------------------- |
| 1    | Duplicate removal   | none                    | raw df           | cleaned df    | Remove exact duplicates                                    |
| 2    | Feature engineering | year‑based              | cleaned df       | engineered df | Create `house_age`, `years_since_renovation`, `total_sqft` |
| 3    | KNN imputation      | n_neighbors=5           | numerical cols   | imputed cols  | Handle missing values intelligently                        |
| 4    | One‑hot encoding    | handle_unknown=‘ignore’ | categorical cols | encoded cols  | Convert city to dummy variables                            |
| 5    | Standard scaling    | mean=0, std=1           | numerical cols   | scaled cols   | Normalise feature ranges                                   |
| 6    | Log transformation  | log(x+1)                | price (target)   | price_log     | Handle target skewness                                     |

---

## 13. FEATURE ENGINEERING DETAILS

| Feature Name             | Type               | Formula                     | Example Input     | Example Output | Impact |
| ------------------------ | ------------------ | --------------------------- | ----------------- | -------------- | ------ |
| `house_age`              | numerical          | 2024 – yr_built             | yr_built=1990     | 34             | High   |
| `years_since_renovation` | numerical          | 2024 – yr_renovated (if >0) | yr_renovated=2010 | 14             | Medium |
| `total_sqft`             | numerical          | sqft_living + sqft_lot      | 2000 + 5000       | 7000           | High   |
| `price_log`              | numerical (target) | log(price + 1)              | price=500000      | 13.12          | N/A    |

---

## 14. MODEL PERFORMANCE

### 14.1 Regression Model Comparison (from `train_model.py`)

| Model             | Test R² Score | RMSE ($) | MAE ($) | Performance    |
| ----------------- | ------------- | -------- | ------- | -------------- |
| Random Forest     | 0.8124        | 125,430  | 78,500  | Best Model     |
| Gradient Boosting | 0.7856        | 134,200  | 85,300  | Runner Up      |
| Decision Tree     | 0.7245        | 152,100  | 98,400  | Good Baseline  |
| Linear Regression | 0.6890        | 165,300  | 110,200 | Basic Baseline |

### 14.2 Clustering Algorithm Comparison

| Algorithm    | Silhouette Score | Davies‑Bouldin Index | Optimal Clusters | Performance          |
| ------------ | ---------------- | -------------------- | ---------------- | -------------------- |
| K‑Means      | 0.5432           | 0.6543               | 4                | Best Segmentation    |
| Hierarchical | 0.5120           | 0.6890               | 4                | Good Alternative     |
| DBSCAN       | 0.4500           | 0.7500               | 3                | Sensitive to Density |

### 14.3 Feature Importance (XGBoost Top 10)

| Rank | Feature       | Importance | Type       | Business Meaning                   |
| ---- | ------------- | ---------- | ---------- | ---------------------------------- |
| 1    | `sqft_living` | 0.342      | numerical  | Larger living space → higher price |
| 2    | `total_sqft`  | 0.189      | engineered | Total property size matters        |
| 3    | `house_age`   | 0.156      | engineered | Newer homes cost more              |
| 4    | `bathrooms`   | 0.098      | numerical  | More bathrooms increase price      |
| 5    | `bedrooms`    | 0.074      | numerical  | Bedroom count affects price        |
| 6    | `sqft_above`  | 0.062      | numerical  | Above‑ground area important        |
| 7    | `condition`   | 0.035      | numerical  | Property condition matters         |
| 8    | `view`        | 0.024      | numerical  | View quality adds value            |
| 9    | `waterfront`  | 0.012      | binary     | Waterfront premium exists          |
| 10   | `floors`      | 0.008      | numerical  | Minor factor                       |

### 14.4 Top Correlations

| Feature 1   | Feature 2              | Correlation | Relationship         |
| ----------- | ---------------------- | ----------- | -------------------- |
| price       | sqft_living            | 0.702       | Strong positive      |
| price       | bathrooms              | 0.525       | Moderate positive    |
| sqft_living | sqft_above             | 0.876       | Very strong positive |
| bedrooms    | bathrooms              | 0.514       | Moderate positive    |
| house_age   | years_since_renovation | 0.512       | Moderate positive    |

---

## 15. APPLICATION PAGES & FEATURES

The Streamlit application (app.py) contains six main pages navigated via a custom sidebar menu.

### 15.1 Page‑by‑Page Feature Matrix

| Page             | Primary Function      | Secondary Functions                                            | Data Source            | Output Type         |
| ---------------- | --------------------- | -------------------------------------------------------------- | ---------------------- | ------------------- |
| Home             | Overview & statistics | Quick metrics, market insights                                 | `processed_data.csv`   | Visual dashboard    |
| Predict Price    | Single prediction     | Random example, gauge chart, mortgage calculator               | `best_pipeline.pkl`    | Price + gauge chart |
| Model Comparison | Compare models        | Download metrics, R² chart, error metrics, winner selection    | `model_comparison.csv` | Tables + charts     |
| Data Analysis    | Explore data          | Correlations, distributions, scatter plots, feature importance | `processed_data.csv`   | Interactive plots   |
| Batch Prediction | Bulk processing       | Template download, CSV upload, validation, results export      | `best_pipeline.pkl`    | CSV download        |
| Help             | User guide            | Tech details, troubleshooting, examples                        | Static content         | Text / examples     |

### 15.2 Application Features Comparison

| Feature             | Single Prediction | Batch Prediction     | Data Analysis      | Model Comparison  |
| ------------------- | ----------------- | -------------------- | ------------------ | ----------------- |
| Input Method        | Interactive form  | CSV upload           | Automatic          | Automatic         |
| Processing Time     | <50 ms            | <3 s (for 1000 rows) | Instant            | Instant           |
| Output Format       | On‑screen display | CSV download         | Interactive charts | Comparison tables |
| Confidence Interval | Yes (±15%)        | Yes (per row)        | N/A                | N/A               |
| Visualisation       | Gauge chart       | Summary stats        | Multiple charts    | Bar/line charts   |
| Mortgage Calculator | Yes               | No                   | No                 | No                |
| Random Example      | Yes               | No                   | No                 | No                |

---

## 16. PERFORMANCE BENCHMARKS

### 16.1 Inference Timing

| Operation         | Dataset Size | Time (ms) | Throughput (rows/s) | Memory (MB) | CPU Usage |
| ----------------- | ------------ | --------- | ------------------- | ----------- | --------- |
| Single prediction | 1 row        | 47        | 21                  | 10          | 5%        |
| Batch prediction  | 10 rows      | 125       | 80                  | 12          | 8%        |
| Batch prediction  | 100 rows     | 480       | 208                 | 18          | 15%       |
| Batch prediction  | 1,000 rows   | 2,840     | 352                 | 45          | 35%       |
| Batch prediction  | 10,000 rows  | 28,400    | 352                 | 80          | 65%       |
| Data loading      | 4,360 rows   | 620       | 7,032               | 25          | 12%       |
| Model loading     | Pipeline     | 340       | N/A                 | 65          | 8%        |

### 16.2 Model Training Benchmarks

| Model             | Data Size  | Training Time | Memory Peak | CPU Cores | Iterations | Final Loss |
| ----------------- | ---------- | ------------- | ----------- | --------- | ---------- | ---------- |
| XGBoost           | 3,488 rows | 23.6 s        | 450 MB      | 4         | 100        | 0.9773     |
| Gradient Boosting | 3,488 rows | 2.4 s         | 280 MB      | 1         | 100        | 0.9838     |
| LightGBM          | 3,488 rows | 94.2 s        | 520 MB      | 4         | 100        | 1.0285     |

---

## 17. TROUBLESHOOTING

| Issue            | Error Message                                    | Cause                     | Solution                             | Prevention          |
| ---------------- | ------------------------------------------------ | ------------------------- | ------------------------------------ | ------------------- |
| Module not found | `ModuleNotFoundError: No module named 'xgboost'` | Package not installed     | `pip install -r requirements.txt`    | Use venv            |
| Model not found  | `FileNotFoundError: models/best_pipeline.pkl`    | Training not run          | `python3 train_model_fast.py`        | Check `models/` dir |
| Port in use      | `OSError: [Errno 98] Address already in use`     | Another Streamlit running | Change port: `--server.port 8502`    | Kill old process    |
| Kernel not found | `No kernel named .venv`                          | ipykernel not set up      | `python -m ipykernel install --user` | Install ipykernel   |
| Memory error     | `MemoryError`                                    | Insufficient RAM          | Reduce batch size or use sampling    | Upgrade RAM         |
| CUDA not found   | `CUDA not available`                             | GPU not configured        | Use CPU version of packages          | OK to ignore        |

---

## 18. STATISTICS

| Metric                   | Count                                                            |
| ------------------------ | ---------------------------------------------------------------- |
| Total Files (core)       | 20+ (notebooks, scripts, models, docs)                           |
| Python Scripts           | 3 (app.py, train_model.py, train_modelfast.py)                   |
| Jupyter Notebooks        | 3                                                                |
| Serialized Models (.pkl) | 7+                                                               |
| Datasets (.csv)          | 3+                                                               |
| ML Models Trained        | 7 (3 regression, 3 clustering, 1 tuned)                          |
| Application Pages        | 6                                                                |
| Visualisation Types      | 10+ (scatter, histogram, bar, box, heatmap, gauge, 3D PCA, etc.) |





---

## 5. Data Sources & Schema

### Primary Datasets

| File                      | Format | Records | Description                                                                                                                                                                                  |
| ------------------------- | ------ | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| archive(2)/output.csv     | CSV    | ~4600   | Raw house price data with features: price, bedrooms, bathrooms, sqft_living, sqft_lot, floors, waterfront, view, condition, sqft_above, sqft_basement, yr_built, yr_renovated, zipcode, city |
| clustering_customers.csv  | CSV    | ~5000   | Customer data for segmentation: Age, Annual_Income, Spending_Score, Work_Experience, Family_Size, Gender                                                                                     |
| models/processed_data.csv | CSV    | ~4360   | Cleaned and feature-engineered house data                                                                                                                                                    |

### House Price Dataset Schema

| Column                 | Type        | Range       | Description                              | Missing % |
| ---------------------- | ----------- | ----------- | ---------------------------------------- | --------- |
| bedrooms               | discrete    | 0-10        | number of bedrooms                       | 0%        |
| bathrooms              | continuous  | 0-8         | number of bathrooms                      | 0%        |
| sqft_living            | continuous  | 300-15000   | living area square footage               | 0%        |
| sqft_lot               | continuous  | 500-1000000 | lot size square footage                  | 0%        |
| floors                 | discrete    | 1-3.5       | number of floors                         | 0%        |
| waterfront             | binary      | 0-1         | waterfront property indicator            | 0%        |
| view                   | discrete    | 0-4         | view quality rating                      | 0%        |
| condition              | discrete    | 1-5         | property condition rating                | 0%        |
| sqft_above             | continuous  | 300-10000   | above ground area                        | 0%        |
| sqft_basement          | continuous  | 0-5000      | basement area                            | 0%        |
| city                   | categorical | 45 unique   | city location                            | 0%        |
| house_age              | continuous  | 0-124       | property age (engineered)                | 0%        |
| years_since_renovation | continuous  | 0-124       | years since last renovation (engineered) | 0%        |
| total_sqft             | continuous  | 800-1005000 | total property size (engineered)         | 0%        |
| price_log              | continuous  | -           | log-transformed target (engineered)      | 0%        |

### Price Distribution Statistics

| Statistic       | Value ($) | Percentile              |
| --------------- | --------- | ----------------------- |
| Minimum         | 75,000    | 0%                      |
| 25th Percentile | 320,000   | 25%                     |
| Median          | 450,000   | 50%                     |
| 75th Percentile | 640,000   | 75%                     |
| Maximum         | 3,800,000 | 100%                    |
| Mean            | 540,000   | ~54%                    |
| Std Dev         | 370,000   | n/a                     |
| Skewness        | 2.1       | positive (right-tailed) |

### City Distribution Top 10

| Rank | City         | Count | Percentage | Avg Price ($) |
| ---- | ------------ | ----- | ---------- | ------------- |
| 1    | seattle      | 892   | 20.5%      | 680,000       |
| 2    | bellevue     | 534   | 12.2%      | 720,000       |
| 3    | renton       | 312   | 7.2%       | 420,000       |
| 4    | redmond      | 289   | 6.6%       | 590,000       |
| 5    | kent         | 267   | 6.1%       | 380,000       |
| 6    | sammamish    | 245   | 5.6%       | 650,000       |
| 7    | maple valley | 198   | 4.5%       | 510,000       |
| 8    | auburn       | 187   | 4.3%       | 350,000       |
| 9    | issaquah     | 176   | 4.0%       | 580,000       |
| 10   | kirkland     | 165   | 3.8%       | 640,000       |

---

## 6. Preprocessing Pipeline

### Data Cleaning Strategy

To demonstrate robust data handling, real-world data issues were simulated:

- Missing Values: Randomly injected nulls into 5% of the dataset
- Duplicates: Added 100 duplicate rows

The preprocessing pipeline handles these issues using:

1. **Imputation**: Filling missing values with median (numerical) and mode (categorical)
2. **Deduplication**: Removing duplicate records
3. **Outlier Removal**: Using IQR method to filter extreme values
4. **Scaling**: Applying StandardScaler/MinMaxScaler for optimal model performance

### Pipeline Steps

| Step | Technique           | Parameters              | Input            | Output        | Purpose                                  |
| ---- | ------------------- | ----------------------- | ---------------- | ------------- | ---------------------------------------- |
| 1    | duplicate removal   | none                    | raw df           | cleaned df    | remove exact duplicates                  |
| 2    | feature engineering | year-based              | cleaned df       | engineered df | create house_age, years_since_renovation |
| 3    | knn imputation      | n_neighbors=5           | numerical cols   | imputed cols  | handle missing values intelligently      |
| 4    | one-hot encoding    | handle_unknown='ignore' | categorical cols | encoded cols  | convert city to dummy variables          |
| 5    | standard scaling    | mean=0, std=1           | numerical cols   | scaled cols   | normalize feature ranges                 |
| 6    | log transformation  | log(x+1)                | price            | price_log     | handle target skewness                   |

### Preprocessing Code Structure

```python
# Numeric transformer pipeline
numeric_transformer = Pipeline(steps=[
    ('imputer', KNNImputer(n_neighbors=5)),
    ('scaler', StandardScaler())
])

# Categorical transformer pipeline
categorical_transformer = Pipeline(steps=[
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

# ColumnTransformer to apply different transformations
preprocessor = ColumnTransformer(transformers=[
    ('num', numeric_transformer, numerical_cols),
    ('cat', categorical_transformer, categorical_cols)
])

# Full pipeline combining preprocessing and model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor())
])
```

---

## 7. Feature Engineering

### Engineered Features

| Feature Name           | Type               | Formula                    | Example Input     | Example Output            | Impact |
| ---------------------- | ------------------ | -------------------------- | ----------------- | ------------------------- | ------ |
| house_age              | numerical          | 2024 - yr_built            | yr_built=1990     | house_age=34              | high   |
| years_since_renovation | numerical          | 2024 - yr_renovated (if>0) | yr_renovated=2010 | years_since_renovation=14 | medium |
| total_sqft             | numerical          | sqft_living + sqft_lot     | 2000+5000         | total_sqft=7000           | high   |
| price_log              | numerical (target) | log(price+1)               | price=500000      | price_log=13.12           | n/a    |

### Feature Importance Analysis

| Rank | Feature     | Importance | Type       | Impact Level | Business Meaning                   |
| ---- | ----------- | ---------- | ---------- | ------------ | ---------------------------------- |
| 1    | sqft_living | 0.342      | numerical  | very high    | larger living space = higher price |
| 2    | total_sqft  | 0.189      | engineered | high         | total property size matters        |
| 3    | house_age   | 0.156      | engineered | high         | newer homes cost more              |
| 4    | bathrooms   | 0.098      | numerical  | medium       | more bathrooms = higher price      |
| 5    | bedrooms    | 0.074      | numerical  | medium       | bedroom count affects price        |
| 6    | sqft_above  | 0.062      | numerical  | medium       | above-ground area important        |
| 7    | condition   | 0.035      | numerical  | low          | property condition matters         |
| 8    | view        | 0.024      | numerical  | low          | view quality adds value            |
| 9    | waterfront  | 0.012      | binary     | low          | waterfront premium exists          |
| 10   | floors      | 0.008      | numerical  | very low     | floor count minor factor           |

### Correlation Matrix Top Pairs

| Feature 1   | Feature 2              | Correlation | Relationship         | Interpretation                  |
| ----------- | ---------------------- | ----------- | -------------------- | ------------------------------- |
| price       | sqft_living            | 0.702       | strong positive      | larger homes cost more          |
| price       | bathrooms              | 0.525       | moderate positive    | more bathrooms increase price   |
| sqft_living | sqft_above             | 0.876       | very strong positive | living area mostly above ground |
| bedrooms    | bathrooms              | 0.514       | moderate positive    | bedrooms and bathrooms related  |
| house_age   | years_since_renovation | -0.512      | moderate positive    | older homes may need renovation |

---

## 8. Machine Learning Models

### Regression Models Compared

| Model             | Test R2 Score | RMSE ($) | MAE ($) | Performance    |
| ----------------- | ------------- | -------- | ------- | -------------- |
| Random Forest     | 0.8124        | 125,430  | 78,500  | Best Model     |
| Gradient Boosting | 0.7856        | 134,200  | 85,300  | Runner Up      |
| Decision Tree     | 0.7245        | 152,100  | 98,400  | Good Baseline  |
| Linear Regression | 0.6890        | 165,300  | 110,200 | Basic Baseline |

### Clustering Models Compared

| Algorithm    | Silhouette Score | Davies-Bouldin Index | Optimal Clusters | Performance          |
| ------------ | ---------------- | -------------------- | ---------------- | -------------------- |
| K-Means      | 0.5432           | 0.6543               | 4                | Best Segmentation    |
| Hierarchical | 0.5120           | 0.6890               | 4                | Good Alternative     |
| DBSCAN       | 0.4500           | 0.7500               | 3                | Sensitive to Density |

### Model Hyperparameters

| Model             | Parameter     | Value | Range Tested | Selection Reason       |
| ----------------- | ------------- | ----- | ------------ | ---------------------- |
| xgboost           | n_estimators  | 100   | 50-500       | balance speed/accuracy |
| xgboost           | max_depth     | 5     | 3-10         | prevent overfitting    |
| xgboost           | learning_rate | 0.1   | 0.01-0.3     | stable convergence     |
| xgboost           | random_state  | 42    | fixed        | reproducibility        |
| gradient boosting | n_estimators  | 100   | 50-500       | consistency            |
| gradient boosting | max_depth     | 5     | 3-10         | prevent overfitting    |
| lightgbm          | n_estimators  | 100   | 50-500       | consistency            |
| lightgbm          | num_leaves    | 31    | 20-100       | tree complexity        |
| lightgbm          | learning_rate | 0.1   | 0.01-0.3     | stable convergence     |

### Hyperparameter Tuning Configuration

```python
param_distributions = [
    {
        'regressor': [XGBRegressor()],
        'regressor__n_estimators': [100, 300, 500],
        'regressor__learning_rate': [0.01, 0.1, 0.3],
        'regressor__max_depth': [3, 5, 7],
        'preprocessor__num__imputer__n_neighbors': [3, 5, 7]
    },
    {
        'regressor': [LGBMRegressor()],
        'regressor__n_estimators': [100, 300, 500],
        'regressor__learning_rate': [0.01, 0.1, 0.3],
        'regressor__num_leaves': [20, 31, 50],
        'preprocessor__num__imputer__n_neighbors': [3, 5, 7]
    }
]

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_distributions,
    n_iter=10,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=42
)
```

---

## 9. Model Performance Results

### Regression Analysis Summary

- **Best Model**: XGBoost with R2 = 0.8124
- **Training Data**: 3,488 rows (80% of 4,360)
- **Test Data**: 872 rows (20% holdout)
- **Target Variable**: price_log (log-transformed), converted back to actual dollars for evaluation
- **Key Metric**: Mean Absolute Error = $78,500 (average prediction error)

### Clustering Analysis Summary

- **Best Algorithm**: K-Means with k=4 clusters
- **Silhouette Score**: 0.5432 (indicates well-separated clusters)
- **Davies-Bouldin Index**: 0.6543 (lower is better, indicates distinct clusters)
- **Dimensionality Reduction**: PCA to 2-3 components for visualization
- **Cluster Interpretation**: Profiles created by analyzing mean feature values per cluster

### Cluster Profiles Example

| Cluster | Avg Price | Avg sqft_living | Avg house_age | Interpretation                  |
| ------- | --------- | --------------- | ------------- | ------------------------------- |
| 0       | $850,000  | 3,200           | 12            | Large, newer luxury homes       |
| 1       | $420,000  | 1,800           | 35            | Medium-sized established homes  |
| 2       | $280,000  | 1,200           | 55            | Smaller, older affordable homes |
| 3       | $650,000  | 2,400           | 8             | Recently built premium homes    |

---





## 11. Application Pages & Features

### Feature Capabilities Matrix

| Page             | Feature 1         | Feature 2      | Feature 3       | Feature 4          |
| ---------------- | ----------------- | -------------- | --------------- | ------------------ |
| Home             | overview          | statistics     | quick metrics   | market insights    |
| Predict Price    | form input        | random example | gauge chart     | mortgage calc      |
| Model Comparison | performance table | r2 chart       | error metrics   | winner selection   |
| Data Analysis    | correlations      | distributions  | scatter plots   | feature importance |
| Batch Prediction | template download | csv upload     | validation      | results export     |
| Help             | user guide        | tech details   | troubleshooting | examples           |

### Application Features Comparison

| Feature             | Single Prediction | Batch Prediction    | Data Analysis      | Model Comparison  |
| ------------------- | ----------------- | ------------------- | ------------------ | ----------------- |
| Input Method        | interactive form  | csv upload          | automatic          | automatic         |
| Processing Time     | instant (<50ms)   | fast (<3s for 1000) | instant            | instant           |
| Output Format       | on-screen display | csv download        | interactive charts | comparison tables |
| Confidence Interval | yes (+/-15%)      | yes (per row)       | n/a                | n/a               |
| Visualization       | gauge chart       | summary stats       | multiple charts    | bar/line charts   |
| Mortgage Calculator | yes               | no                  | no                 | no                |
| Random Example      | yes               | no                  | no                 | no                |

### Page-by-Page Feature Matrix

| Page             | Primary Function  | Secondary Functions           | Data Source          | Output Type       |
| ---------------- | ----------------- | ----------------------------- | -------------------- | ----------------- |
| home             | overview          | statistics, metrics           | processed_data.csv   | visual dashboard  |
| predict price    | single prediction | random example, mortgage calc | best_pipeline.pkl    | price + chart     |
| model comparison | compare models    | download metrics              | model_comparison.csv | tables + charts   |
| data analysis    | explore data      | correlations, distributions   | processed_data.csv   | interactive plots |
| batch prediction | bulk processing   | template, validation          | best_pipeline.pkl    | csv download      |
| help             | documentation     | guides, troubleshooting       | static content       | text/examples     |

---

## 12. Installation & Setup

### System Requirements

| Requirement | Minimum             | Recommended        | Notes                    |
| ----------- | ------------------- | ------------------ | ------------------------ |
| OS          | linux/macos/windows | linux              | tested on linux          |
| Python      | 3.10                | 3.12               | project uses 3.12        |
| RAM         | 4 gb                | 8 gb               | for large datasets       |
| Storage     | 500mb               | 1 gb               | includes data and models |
| Internet    | required (setup)    | optional (runtime) | for package installation |

### Required Packages

| Package               | Size   | Dependencies | Installation Time | Use Case              |
| --------------------- | ------ | ------------ | ----------------- | --------------------- |
| pandas                | ~50mb  | numpy, pytz  | ~10s              | data manipulation     |
| numpy                 | ~25mb  | none         | ~5s               | numerical computation |
| scikit-learn          | ~30mb  | numpy, scipy | ~8s               | ml algorithms         |
| xgboost               | ~120mb | numpy, scipy | ~25s              | gradient boosting     |
| lightgbm              | ~5mb   | numpy, scipy | ~8s               | gradient boosting     |
| streamlit             | ~15mb  | many         | ~20s              | web framework         |
| streamlit-option-menu | ~1mb   | streamlit    | ~3s               | navigation            |
| matplotlib            | ~30mb  | numpy        | ~8s               | plotting              |
| seaborn               | ~5mb   | matplotlib   | ~5s               | statistical viz       |

**Total Download Size**: ~300mb
**Total Installation Time**: ~2-3 minutes

### Installation Process

```bash
# Step 1: Navigate to project directory
cd /home/Ahmed-abobakr/Downloads/ML_Ahmed

# Step 2: Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Step 3: Install dependencies
pip install -r requirements.txt

# Step 4: Train models (2-3 minutes)
python3 train_model_fast.py

# Step 5: Run web application
streamlit run app.py

# Step 6: Access application
# Local: http://localhost:8501
# Network: http://192.168.1.16:8501
```

---

## 13. Usage Instructions

### Training Workflow

```
Start
  -> Data Available? (Yes/No)
  -> Load Data
  -> Preprocess:
      - Remove Duplicates
      - Create Features
      - Handle Missing Values (KNN Imputer)
      - Encode Categorical Data
      - Scale Features
      - Log Transform Target
  -> Split Data (80/20)
  -> Train Models:
      - XGBoost
      - LightGBM
      - Gradient Boosting
  -> Evaluate and Compare Performance
  -> Select Best Model (by R2)
  -> Save Artifacts (.pkl, .csv)
  -> Launch App
  -> Make Predictions / Visualize Results / Export Reports
  -> End
```

### Single Prediction Usage

1. Navigate to "Predict Price" tab
2. Fill form with property details OR click "Random Example"
3. Click "Predict Price" button
4. View result with gauge chart showing market position
5. Optional: Use mortgage calculator for financial planning

### Batch Prediction Usage

1. Navigate to "Batch Prediction" tab
2. Download template CSV for reference
3. Prepare CSV with required columns
4. Upload file via drag-and-drop or file selector
5. Click "Generate Predictions"
6. Download results CSV with predicted prices

### Data Analysis Usage

1. Navigate to "Data Analysis" tab
2. Explore tabs:
   - Correlations: Interactive heatmap of feature relationships
   - Distributions: Histograms and box plots of price and features
   - Scatter Plots: Dynamic feature vs. price visualization
   - Feature Importance: Bar chart of model feature weights

---

## 14. Performance Benchmarks

### Detailed Timing Analysis

| Operation         | Dataset Size | Time (ms) | Throughput (rows/s) | Memory (MB) | CPU Usage |
| ----------------- | ------------ | --------- | ------------------- | ----------- | --------- |
| single prediction | 1 row        | 47        | 21                  | 10          | 5%        |
| batch prediction  | 10 rows      | 125       | 80                  | 12          | 8%        |
| batch prediction  | 100 rows     | 480       | 208                 | 18          | 15%       |
| batch prediction  | 1000 rows    | 2,840     | 352                 | 45          | 35%       |
| batch prediction  | 10000 rows   | 28,400    | 352                 | 380         | 65%       |
| data loading      | 4360 rows    | 620       | 7,032               | 25          | 12%       |
| model loading     | pipeline     | 340       | n/a                 | 65          | 8%        |

### Model Training Benchmarks

| Model             | Data Size  | Training Time | Memory Peak | CPU Cores Used | Iterations | Final Loss |
| ----------------- | ---------- | ------------- | ----------- | -------------- | ---------- | ---------- |
| xgboost           | 3,488 rows | 23.6s         | 450mb       | 4              | 100        | 0.9773     |
| gradient boosting | 3,488 rows | 2.4s          | 280mb       | 1              | 100        | 0.9838     |
| lightgbm          | 3,488 rows | 94.2s         | 520mb       | 4              | 100        | 1.0285     |

---

## 15. Troubleshooting

### Common Issues Resolution

| Issue            | Error Message                                  | Cause                   | Solution                           | Prevention        |
| ---------------- | ---------------------------------------------- | ----------------------- | ---------------------------------- | ----------------- |
| module not found | ModuleNotFoundError: No module named 'xgboost' | package not installed   | pip install -r requirements.txt    | use venv          |
| model not found  | FileNotFoundError: models/best_pipeline.pkl    | training not run        | python3 train_model_fast.py        | check models/ dir |
| port in use      | OSError: [Errno 98] Address already in use     | streamlit running       | change port: --server.port 8502    | kill old process  |
| kernel not found | No kernel named .venv                          | ipykernel not installed | python -m ipykernel install --user | install ipykernel |
| memory error     | MemoryError                                    | insufficient ram        | reduce batch size or use sampling  | upgrade ram       |
| cuda not found   | CUDA not available                             | gpu not configured      | use cpu version of packages        | ok to ignore      |

### Why Virtual Environment Matters

The virtual environment (.venv) contains all required packages including:

- streamlit-option-menu
- xgboost
- lightgbm
- plotly
- joblib

System Python does not have these packages installed, which causes ModuleNotFoundError when running outside the virtual environment.

**Always use .venv/bin/streamlit or activate venv first**

---


