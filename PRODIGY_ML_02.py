import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/Users/amaymalviya/Downloads/Mall_Customers.csv'
data = pd.read_csv(file_path)

# Data Preprocessing: No missing values to handle
# Feature Selection
X = data[['Annual Income (k$)', 'Spending Score (1-100)']]

# Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model Training
k = 3  # You can choose the number of clusters
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(X_scaled)
labels = kmeans.labels_
data['cluster'] = labels

# Model Evaluation
score = silhouette_score(X_scaled, labels)
print(f'Silhouette Score: {score}')

# Visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Annual Income (k$)', y='Spending Score (1-100)', hue='cluster', data=data, palette='viridis')
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.show()
