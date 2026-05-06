import nbformat as nbf

def create_notebook(filename, cells_data):
    nb = nbf.v4.new_notebook()
    cells = []
    for cell_type, source in cells_data:
        if cell_type == 'markdown':
            cells.append(nbf.v4.new_markdown_cell(source))
        elif cell_type == 'code':
            cells.append(nbf.v4.new_code_cell(source))
    nb['cells'] = cells
    with open(filename, 'w') as f:
        nbf.write(nb, f)
    print(f"Created {filename}")

# 1. Regression Notebook
regression_cells = [
    ('markdown', """# regression analysis: house price prediction

**5*a by ahmed abobakr**

## project overview
this project aims to predict house prices based on various features such as square footage, number of bedrooms, and location. we will explore the dataset, preprocess the data, and train multiple regression models to find the best predictor.

## objectives
1.  **data exploration**: understand the dataset structure and distributions.
2.  **preprocessing**: clean data, handle missing values, remove outliers, and scale features.
3.  **modeling**: train linear regression, decision tree, random forest, and gradient boosting models.
4.  **evaluation**: compare models using r-squared, rmse, and mae metrics."""),
    ('markdown', "---"),
    ('markdown', """## 1. import libraries
we start by importing the necessary libraries for data manipulation, visualization, and machine learning."""),
    ('code', """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

print("libraries imported successfully")"""),
    ('markdown', "---"),
    ('markdown', """## 2. load dataset
loading the house prices dataset and displaying the first few rows to understand its structure."""),
    ('code', """df = pd.read_csv('archive (2)/output.csv')
print(f"dataset shape: {df.shape}")
df.head()"""),
    ('markdown', """### simulating real-world data issues
since the original dataset is clean, we will introduce some missing values and duplicates to demonstrate the preprocessing capabilities required for this project."""),
    ('code', """# introduce missing values
np.random.seed(42)
mask = np.random.random(df.shape) < 0.05  # 5% missing data
df_dirty = df.mask(mask)

# introduce duplicates
duplicates = df_dirty.sample(n=100, random_state=42)
df = pd.concat([df_dirty, duplicates], ignore_index=True)

print(f"dataset shape after adding issues: {df.shape}")
print(f"missing values count: {df.isnull().sum().sum()}")
print(f"duplicates count: {df.duplicated().sum()}")"""),
    ('markdown', "---"),
    ('markdown', """## 3. exploratory data analysis (eda)
analyzing the data distribution and statistical summaries to identify patterns and potential issues."""),
    ('code', """# display dataset information
df.info()"""),
    ('code', """# statistical summary of numerical features
df.describe()"""),
    ('markdown', """### visualizing price distribution
understanding the target variable (price) distribution is crucial for regression tasks."""),
    ('code', """plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.hist(df['price'].dropna(), bins=50, color='skyblue', edgecolor='black')
plt.title('distribution of house prices')
plt.xlabel('price')
plt.ylabel('frequency')

plt.subplot(1, 2, 2)
plt.boxplot(df['price'].dropna(), vert=True)
plt.title('boxplot of house prices')
plt.ylabel('price')

plt.tight_layout()
plt.show()"""),
    ('markdown', "---"),
    ('markdown', """## 4. data preprocessing
preparing the data for modeling by handling missing values, removing duplicates, handling outliers, and encoding categorical variables."""),
    ('code', """df_processed = df.copy()

# 1. handle missing values
numerical_cols = df_processed.select_dtypes(include=[np.number]).columns
for col in numerical_cols:
    if df_processed[col].isnull().sum() > 0:
        df_processed[col].fillna(df_processed[col].median(), inplace=True)

categorical_cols = df_processed.select_dtypes(include=['object']).columns
for col in categorical_cols:
    if df_processed[col].isnull().sum() > 0:
        df_processed[col].fillna(df_processed[col].mode()[0], inplace=True)

# 2. remove duplicates
df_processed = df_processed.drop_duplicates()

# 3. handle outliers (iqr method)
q1 = df_processed['price'].quantile(0.25)
q3 = df_processed['price'].quantile(0.75)
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
df_processed = df_processed[(df_processed['price'] >= lower_bound) & (df_processed['price'] <= upper_bound)]

# 4. drop irrelevant columns
cols_to_drop = ['date', 'street', 'statezip', 'country']
df_processed = df_processed.drop(columns=[c for c in cols_to_drop if c in df_processed.columns])

# 5. encode categorical variables
if 'city' in df_processed.columns:
    le = LabelEncoder()
    df_processed['city_encoded'] = le.fit_transform(df_processed['city'].astype(str))
    df_processed = df_processed.drop('city', axis=1)

print("preprocessing completed successfully")
print(f"final dataset shape: {df_processed.shape}")"""),
    ('markdown', "---"),
    ('markdown', """## 5. train-test split & scaling
splitting the data into training and testing sets (80/20 split) and scaling features for optimal model performance."""),
    ('code', """X = df_processed.drop('price', axis=1)
y = df_processed['price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("data split and scaled")
print(f"training samples: {X_train.shape[0]}")
print(f"testing samples: {X_test.shape[0]}")"""),
    ('markdown', "---"),
    ('markdown', """## 6. model training & evaluation
training four different regression models and evaluating their performance using r-squared, rmse, and mae."""),
    ('code', """models = {
    'linear regression': LinearRegression(),
    'decision tree': DecisionTreeRegressor(max_depth=10, random_state=42),
    'random forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    'gradient boosting': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
}

results = []

print("training models...")
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    results.append({'model': name, 'r2': r2, 'rmse': rmse, 'mae': mae})
    print(f" - {name}: r2={r2:.4f}")"""),
    ('markdown', "---"),
    ('markdown', """## 7. results comparison
comparing the performance of all trained models to identify the best one."""),
    ('code', """results_df = pd.DataFrame(results).sort_values('r2', ascending=False)
print("\\nmodel performance comparison:")
print(results_df)

plt.figure(figsize=(10, 6))
sns.barplot(x='r2', y='model', data=results_df, palette='viridis')
plt.title('model comparison - r2 score')
plt.xlabel('r-squared score')
plt.ylabel('model')
plt.xlim(0, 1)
plt.show()"""),
    ('markdown', """## conclusion

**5*a by ahmed abobakr**

the regression analysis is complete. the model with the highest r-squared score is considered the best for predicting house prices in this dataset.""")
]

create_notebook('01_Regression_House_Prices.ipynb', regression_cells)

# 2. Customer Segmentation Notebook
customer_cells = [
    ('markdown', """# clustering analysis: customer segmentation

**5*a by ahmed abobakr**

## project overview
this project focuses on segmenting customers based on their behavior and demographics. using clustering algorithms, we aim to identify distinct groups of customers to enable targeted marketing strategies.

## objectives
1.  **data preparation**: load and clean the customer dataset.
2.  **preprocessing**: encode categorical variables and scale features.
3.  **optimal k**: determine the best number of clusters using the elbow method.
4.  **modeling**: apply k-means clustering to segment customers.
5.  **evaluation**: analyze the clusters and visualize the results."""),
    ('markdown', "---"),
    ('markdown', """## 1. import libraries
importing necessary libraries for data analysis, visualization, and clustering."""),
    ('code', """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.cluster.hierarchy import dendrogram, linkage
import warnings
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette('husl')

print("libraries imported successfully")"""),
    ('markdown', "---"),
    ('markdown', """## 2. load dataset
loading the customer segmentation data."""),
    ('code', """df = pd.read_csv('clustering_customers.csv')
print(f"dataset shape: {df.shape}")
df.head()"""),
    ('markdown', """### simulating real-world data issues
since the original dataset is clean, we will introduce some missing values and duplicates to demonstrate the preprocessing capabilities required for this project."""),
    ('code', """# introduce missing values
np.random.seed(42)
mask = np.random.random(df.shape) < 0.05  # 5% missing data
df_dirty = df.mask(mask)

# introduce duplicates
duplicates = df_dirty.sample(n=100, random_state=42)
df = pd.concat([df_dirty, duplicates], ignore_index=True)

print(f"dataset shape after adding issues: {df.shape}")
print(f"missing values count: {df.isnull().sum().sum()}")
print(f"duplicates count: {df.duplicated().sum()}")"""),
    ('markdown', "---"),
    ('markdown', """## 3. data preprocessing
cleaning the data by removing nulls and duplicates, encoding gender, and scaling numerical features."""),
    ('code', """# 1. clean data
df = df.dropna()
df = df.drop_duplicates()
print(f"shape after cleaning: {df.shape}")

# 2. encode gender
le = LabelEncoder()
if 'Gender' in df.columns:
    df['Gender'] = le.fit_transform(df['Gender'])

# 3. select features and scale
features = ['Age', 'Annual_Income', 'Spending_Score', 'Work_Experience', 'Family_Size']
features = [col for col in features if col in df.columns]
X = df[features]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=features)

print("data preprocessing completed")"""),
    ('markdown', "---"),
    ('markdown', """## 4. determine optimal clusters
using the elbow method to find the optimal number of clusters (k)."""),
    ('code', """wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 5))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('elbow method for optimal k')
plt.xlabel('number of clusters')
plt.ylabel('wcss (within-cluster sum of squares)')
plt.show()"""),
    ('markdown', "---"),
    ('markdown', """## 5. model building: k-means clustering
applying k-means with the chosen number of clusters (k=4)."""),
    ('code', """k = 4
kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

print(f"k-means clustering completed with k={k}")"""),
    ('markdown', "---"),
    ('markdown', """## 6. evaluation & visualization
evaluating the model using silhouette score and visualizing the customer segments."""),
    ('code', """score = silhouette_score(X_scaled, clusters)
print(f"silhouette score: {score:.4f}")

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='Annual_Income', y='Spending_Score', hue='Cluster', palette='viridis', s=100, alpha=0.7)
plt.title('customer segments: income vs spending score')
plt.xlabel('annual income')
plt.ylabel('spending score')
plt.legend(title='cluster')
plt.show()"""),
    ('markdown', """## conclusion

**5*a by ahmed abobakr**

we have successfully segmented the customers into distinct groups based on their income and spending habits. this information can be valuable for targeted marketing campaigns.""")
]

create_notebook('02_Clustering_Customer_Segmentation.ipynb', customer_cells)

# 3. House Price Clustering Notebook
house_clustering_cells = [
    ('markdown', """# house price clustering analysis with pca

**5*a by ahmed abobakr**

## project overview
this notebook demonstrates advanced clustering techniques to segment houses based on their features. we will use dimensionality reduction (pca & t-sne) and multiple clustering algorithms (k-means, dbscan, hierarchical) to identify distinct property groups.

## objectives
1.  **data preparation**: load and scale the data for clustering.
2.  **dimensionality reduction**: use pca and t-sne to visualize high-dimensional data.
3.  **optimal k**: determine the best number of clusters.
4.  **modeling**: apply k-means, dbscan, and hierarchical clustering.
5.  **evaluation**: compare models using silhouette score and davies-bouldin index.
6.  **profiling**: analyze and interpret the characteristics of each cluster."""),
    ('markdown', "---"),
    ('markdown', """## 1. import libraries & load data
importing necessary libraries and loading the preprocessed dataset."""),
    ('code', """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("libraries imported successfully")"""),
    ('code', """# load data
df = pd.read_csv('models/processed_data.csv')
print(f"dataset shape: {df.shape}")
df.head()"""),
    ('markdown', "---"),
    ('markdown', """## 2. data preparation
selecting numerical features and scaling them for clustering."""),
    ('code', """# select features for clustering (numerical only)
feature_cols = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors', 
                'waterfront', 'view', 'condition', 'sqft_above', 'sqft_basement',
                'house_age', 'years_since_renovation', 'total_sqft']

X = df[feature_cols].copy()

# handle any missing values
X = X.fillna(X.median())

# scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("data scaled successfully")"""),
    ('markdown', "---"),
    ('markdown', """## 3. dimensionality reduction
reducing dimensions using pca and t-sne to visualize the data structure in 2d."""),
    ('code', """# pca
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print(f"pca explained variance: {pca.explained_variance_ratio_.sum():.2%}")

# t-sne (using a subset for speed)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled[:1000])
print("t-sne transformation complete")"""),
    ('markdown', "---"),
    ('markdown', """## 4. determine optimal clusters
using the elbow method and silhouette scores to find the optimal number of clusters."""),
    ('code', """inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, kmeans.labels_))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(K_range, inertias, 'bo-')
ax1.set_title('elbow method')
ax1.set_xlabel('number of clusters')
ax1.set_ylabel('inertia')

ax2.plot(K_range, silhouette_scores, 'ro-')
ax2.set_title('silhouette score')
ax2.set_xlabel('number of clusters')
ax2.set_ylabel('score')

plt.show()
print(f"best k by silhouette: {K_range[np.argmax(silhouette_scores)]}")"""),
    ('markdown', "---"),
    ('markdown', """## 5. clustering modeling
applying k-means, dbscan, and hierarchical clustering algorithms."""),
    ('code', """optimal_k = 4

# 1. k-means
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
labels_kmeans = kmeans.fit_predict(X_scaled)

# 2. dbscan
dbscan = DBSCAN(eps=3, min_samples=10)
labels_dbscan = dbscan.fit_predict(X_scaled)

# 3. hierarchical
hierarch = AgglomerativeClustering(n_clusters=optimal_k)
labels_hierarch = hierarch.fit_predict(X_scaled)

print("clustering completed")"""),
    ('markdown', "---"),
    ('markdown', """## 6. evaluation & comparison
comparing the performance of the three algorithms."""),
    ('code', """results = {
    'k-means': {
        'silhouette': silhouette_score(X_scaled, labels_kmeans),
        'davies-bouldin': davies_bouldin_score(X_scaled, labels_kmeans),
        'n_clusters': len(np.unique(labels_kmeans))
    },
    'dbscan': {
        'silhouette': silhouette_score(X_scaled, labels_dbscan) if len(np.unique(labels_dbscan)) > 1 else -1,
        'davies-bouldin': davies_bouldin_score(X_scaled, labels_dbscan) if len(np.unique(labels_dbscan)) > 1 else -1,
        'n_clusters': len(np.unique(labels_dbscan[labels_dbscan != -1]))
    },
    'hierarchical': {
        'silhouette': silhouette_score(X_scaled, labels_hierarch),
        'davies-bouldin': davies_bouldin_score(X_scaled, labels_hierarch),
        'n_clusters': len(np.unique(labels_hierarch))
    }
}

results_df = pd.DataFrame(results).T
print("clustering algorithm comparison:")
print(results_df)"""),
    ('markdown', "---"),
    ('markdown', """## 7. cluster profiling
analyzing the characteristics of each cluster identified by k-means."""),
    ('code', """df_clustered = df[feature_cols].copy()
df_clustered['cluster'] = labels_kmeans
df_clustered['price'] = np.expm1(df['price_log'])

cluster_profiles = df_clustered.groupby('cluster').mean()
print("cluster profiles (average values):")
print(cluster_profiles.round(2))

# visualize
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
cluster_profiles['price'].plot(kind='bar', ax=axes[0,0], color='skyblue', title='avg price')
cluster_profiles['sqft_living'].plot(kind='bar', ax=axes[0,1], color='lightgreen', title='avg living area')
cluster_profiles['bedrooms'].plot(kind='bar', ax=axes[1,0], color='lightcoral', title='avg bedrooms')
cluster_profiles['house_age'].plot(kind='bar', ax=axes[1,1], color='gold', title='avg house age')
plt.tight_layout()
plt.show()"""),
    ('markdown', """## conclusion

**5*a by ahmed abobakr**

we successfully identified distinct segments of houses. these clusters can help in understanding the market structure and pricing strategies.""")
]

create_notebook('02_Clustering_House_Prices.ipynb', house_clustering_cells)
