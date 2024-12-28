import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import io
import base64
from sklearn.decomposition import PCA

app = Flask(__name__)
df = pd.read_csv('user_behavior_dataset.csv')

@app.route('/')
def home():
    print("Flask app running...")
    return render_template('index.html', title="Home", header="Welcome to Flask")

@app.route('/about')
def about():
    print("Flask app running...")
    return render_template('base.html', title="About", header="About Flask")

@app.route('/clustering')
def cluster():
    columns_to_transform = ['App Usage Time (min/day)', 'Screen On Time (hours/day)',
                            'Battery Drain (mAh/day)', 'Data Usage (MB/day)']
    numeric_columns = columns_to_transform + ['Number of Apps Installed', 'Age']

    # 1. Logarithmic Transformation
    for col in columns_to_transform:
        df[f'{col}_log'] = np.log1p(df[col])  # log1p handles zero values

    # 2. Scaling with StandardScaler
    scaler = StandardScaler()
    df_scaled = pd.DataFrame(
        scaler.fit_transform(df[[f'{col}_log' for col in columns_to_transform] + ['Number of Apps Installed', 'Age']]),
        columns=[f'{col}_log' for col in columns_to_transform] + ['Number of Apps Installed', 'Age'])

    # 3. (Optional) Dimensionality Reduction with PCA or UMAP
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df_scaled)
    df['PCA1'], df['PCA2'] = X_pca[:, 0], X_pca[:, 1]
    X_normalized = df_scaled  # Menggunakan data yang sudah dinormalisasi

    # 2. Determine the Optimal Number of Clusters using the Elbow Method
    sse = []  # Sum of Squared Errors
    max_clusters = 10

    for n_clusters in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X_normalized)
        sse.append(kmeans.inertia_)

    # Plot Elbow Method
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_clusters + 1), sse, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.grid()

    # Save the plot to a BytesIO object and encode it as base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    # Apply K-Means Clustering with Optimal Number of Clusters (e.g., k=3)
    optimal_clusters = 5  # Pilih berdasarkan elbow method
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    df_scaled['Cluster'] = kmeans.fit_predict(X_normalized)

    # Visualize the Clusters (Optional, using PCA for 2D visualization)
    plt.figure(figsize=(8, 5))
    plt.scatter(df['PCA1'], df['PCA2'], c=df_scaled['Cluster'], cmap='viridis', s=50)
    plt.title('K-Means Clustering Results')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.colorbar(label='Cluster')
    plt.grid()

    # Save the plot to a BytesIO object and encode it as base64 for cluster plot
    img2 = io.BytesIO()
    plt.savefig(img2, format='png')
    img2.seek(0)
    cluster_plot_url = base64.b64encode(img2.getvalue()).decode()

    # 5. Display Clustered Data (Optional)
    print(df_scaled.head())

    return render_template('clustering.html', title="Cluster", header="Clustering",
                           plot_url=plot_url, cluster_plot_url=cluster_plot_url)


if __name__ == '__main__':
    app.run(debug=True)
