import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import io
import base64
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score

app = Flask(__name__)
df = pd.read_csv('user_behavior_dataset.csv')

@app.route('/layout')
def layout():
    return render_template('layout.html')

@app.route('/')
def home():
    print("Flask app running...")
    return render_template('index.html', title="Home", header="Welcome to Flask")

@app.route('/about')
def about():
    print("Flask app running...")
    return render_template('base.html', title="About", header="About Flask")

@app.route('/cluster')
def cluster():
    # Nentuin kolom yang perlu dinormalisasi min max
    columns_to_transform = ['App Usage Time (min/day)', 'Screen On Time (hours/day)',
                            'Battery Drain (mAh/day)', 'Data Usage (MB/day)',
                            'Number of Apps Installed', 'Age']

    # Min-max normalisasi dari skala 1-10
    scaler = MinMaxScaler(feature_range=(1, 10))
    df_normalized = pd.DataFrame(
        scaler.fit_transform(df[columns_to_transform]),
        columns=columns_to_transform
    )

    # Mencari jumlah cluster optimal pakai elbow method
    sse = []  # Sum of Squared Errors
    max_clusters = 10

    for n_clusters in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(df_normalized)
        sse.append(kmeans.inertia_)

    # Plotting Elbow Method
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_clusters + 1, 1), sse, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Sum of Squared Errors (SSE)')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.grid(True)

    # Simpen plot ke objek BytesIO dan encode ke base64
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()

    # K-means clustering dengan 5 cluster
    optimal_clusters = 5
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    df_normalized['Cluster'] = kmeans.fit_predict(df_normalized)

    # Evaluasi pakai silhouette score
    silhouette_avg = silhouette_score(df_normalized, df_normalized['Cluster'])

    # Reduksi dimensi dengan pca untuk visualisasi
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(df_normalized)
    df_normalized['PCA1'], df_normalized['PCA2'] = X_pca[:, 0], X_pca[:, 1]

    # Visualisasi clustering pakai scatter plot
    plt.figure(figsize=(8, 5))
    scatter = plt.scatter(df_normalized['PCA1'], df_normalized['PCA2'],
                          c=df_normalized['Cluster'], cmap='viridis', s=50)
    plt.title('K-Means Clustering Results')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.grid(True)

    # Simpen scatter plot ke objek BytesIO dan encode ke base64
    img2 = io.BytesIO()
    plt.savefig(img2, format='png')
    img2.seek(0)
    cluster_plot_url = base64.b64encode(img2.getvalue()).decode()
    plt.close()

    # Menyusun hasil evaluasi dan plot untuk dikirim ke template
    evaluation = {
        'Silhouette Score': silhouette_avg,
        'SSE': sse
    }

    return render_template('cluster.html',
                           title="Clustering",
                           header="Clustering",
                           plot_url=plot_url,
                           cluster_plot_url=cluster_plot_url,
                           evaluation=evaluation,
                           optimal_clusters=optimal_clusters)


@app.route('/classification')
def classification():
    try:
        
        df_cleaned = df.fillna(df.median())  # Mengisi nilai yang hilang dengan median


        # contoh, memprediksi pengguna dengan "Hight Data Usage"
        # Membuat kolom target baru berdasarkan threshold data usage
        if 'High Data Usage' not in df.columns:
            df['High Data Usage'] = (df['Data Usage (MB/day)'] > 1000).astype(int)  # 1 jika > 1000 MB, 0 jika ≤ 1000 MB

        # Pisahkan X (fitur) dan y (target)
        X = df.drop('High Data Usage', axis=1)
        y = df['High Data Usage']


        # Membagi data menjadi train dan test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Decision Tree Model
        model = DecisionTreeClassifier(random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluation Metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        # Plot Confusion Matrix
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        img = io.BytesIO()
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()
        plt.close()

        # Mengirimkan hasil ke template HTML
        return render_template('classification.html',
                               title="Classification",
                               header="Decision Tree Classification",
                               report=report,
                               plot_url=plot_url)

    except Exception as e:
        return f"An error occurred: {str(e)}", 500


if __name__ == '__main__':
    app.run(debug=True)
