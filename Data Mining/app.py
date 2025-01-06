import numpy as np
from flask import Flask, request, jsonify, render_template
import pandas as pd
from pyexpat import model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.metrics import silhouette_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
from sklearn.ensemble import RandomForestRegressor
import joblib
from sklearn.ensemble import IsolationForest
from mlxtend.frequent_patterns import fpgrowth, apriori
from mlxtend.frequent_patterns import association_rules
import networkx as nx
from sklearn.cluster import KMeans, AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage


app = Flask(__name__)
df = pd.read_csv('user_behavior_dataset.csv')

# Preprocessing data and training model
def train_model():
    # Select relevant columns
    columns_to_use = ['App Usage Time (min/day)', 'Screen On Time (hours/day)',
                      'Data Usage (MB/day)', 'Number of Apps Installed', 'Age']
    X = df[columns_to_use]
    y = df['Battery Drain (mAh/day)']

    # Splitting dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scaling features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save scaler for future use
    joblib.dump(scaler, 'scaler.pkl')

    # Train a RandomForestRegressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    joblib.dump(rf_model, 'rf_model.pkl')

    # Train a LinearRegression model
    lr_model = LinearRegression()
    lr_model.fit(X_train_scaled, y_train)
    joblib.dump(lr_model, 'lr_model.pkl')
    print("Model has been saved as 'model.pkl' and scaler as 'scaler.pkl'")
# Train the model when the app starts
train_model()

@app.route('/layout')
def layout():
    return render_template('layout.html')

@app.route('/')
def home():
    print("Flask app running...")
    return render_template('index.html', title="Home", header="Welcome to Flask", active_page="home")

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
    elbow_img = io.BytesIO()
    plt.savefig(elbow_img, format='png')
    elbow_img.seek(0)
    elbow_plot_url = base64.b64encode(elbow_img.getvalue()).decode()
    plt.close()

    # K-means clustering dengan 5 cluster
    optimal_clusters = 5
    kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
    df_normalized['KMeans_Cluster'] = kmeans.fit_predict(df_normalized)

    # Evaluasi pakai silhouette score
    silhouette_avg_kmeans = silhouette_score(df_normalized, df_normalized['KMeans_Cluster'])

    # Hierarchical Clustering
    hierarchical = AgglomerativeClustering(n_clusters=optimal_clusters, linkage='ward', metric='euclidean')
    df_normalized['Hierarchical_Cluster'] = hierarchical.fit_predict(df_normalized)

    # Evaluasi pakai silhouette score
    silhouette_avg_hierarchical = silhouette_score(df_normalized, df_normalized['Hierarchical_Cluster'])

    # Dendrogram untuk Hierarchical Clustering
    plt.figure(figsize=(10, 7))
    linked = linkage(df_normalized, method='ward')
    dendrogram(linked)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Data Points')
    plt.ylabel('Euclidean Distance')

    dendrogram_img = io.BytesIO()
    plt.savefig(dendrogram_img, format='png')
    dendrogram_img.seek(0)
    dendrogram_plot_url = base64.b64encode(dendrogram_img.getvalue()).decode()
    plt.close()

    # Reduksi dimensi dengan pca untuk visualisasi
    pca = PCA(n_components=2)
    pca_results = pca.fit_transform(df_normalized)
    df_normalized['PCA1'] = pca_results[:, 0]
    df_normalized['PCA2'] = pca_results[:, 1]

    # Scatter plot untuk K-Means
    plt.figure(figsize=(8, 5))
    plt.scatter(df_normalized['PCA1'], df_normalized['PCA2'],
                c=df_normalized['KMeans_Cluster'], cmap='viridis', s=50)
    plt.title('K-Means Clustering Results')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.grid(True)

    kmeans_img = io.BytesIO()
    plt.savefig(kmeans_img, format='png')
    kmeans_img.seek(0)
    kmeans_plot_url = base64.b64encode(kmeans_img.getvalue()).decode()
    plt.close()

    # Scatter plot untuk Hierarchical Clustering
    plt.figure(figsize=(8, 5))
    plt.scatter(df_normalized['PCA1'], df_normalized['PCA2'],
                c=df_normalized['Hierarchical_Cluster'], cmap='viridis', s=50)
    plt.title('Hierarchical Clustering Results')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.grid(True)

    hierarchical_img = io.BytesIO()
    plt.savefig(hierarchical_img, format='png')
    hierarchical_img.seek(0)
    hierarchical_plot_url = base64.b64encode(hierarchical_img.getvalue()).decode()
    plt.close()

    # Hasil evaluasi
    evaluation = {
        'Silhouette Score (K-Means)': silhouette_avg_kmeans,
        'Silhouette Score (Hierarchical)': silhouette_avg_hierarchical,
        'SSE': sse
    }

    return render_template('cluster.html',
                           title="Clustering",
                           header="Clustering Results",
                           elbow_plot_url=elbow_plot_url,
                           kmeans_plot_url=kmeans_plot_url,
                           hierarchical_plot_url=hierarchical_plot_url,
                           dendrogram_plot_url=dendrogram_plot_url,
                           evaluation=evaluation,
                           optimal_clusters=optimal_clusters,
                           active_page="cluster")



@app.route('/classification')
def classification():
    try:
        # Identifikasi kolom string dan kategori
        for column in df.columns:
            if df[column].dtype == 'object' or df[column].dtype == 'category':
                print(f"Encoding column: {column}")
                label_encoder = LabelEncoder()
                df[column] = label_encoder.fit_transform(df[column].astype(str))

        # Pastikan semua kolom numerik
        df_cleaned = df.apply(pd.to_numeric, errors='coerce')

        # Debug: Meriksa dataset setelah encoding
        print(df_cleaned.dtypes)
        print(df_cleaned.head())

        # Membuat kolom target 'High Data Usage'
        if 'High Data Usage' not in df_cleaned.columns:
            df_cleaned['High Data Usage'] = (df_cleaned['Data Usage (MB/day)'] > 1000).astype(int)

        # Pisahkan fitur (X) dan target (y)
        X = df_cleaned.drop(['High Data Usage'], axis=1)
        y = df_cleaned['High Data Usage']

        # Debug: Periksa data input model
        print("Features (X):", X.head())
        print("Target (y):", y.head())

        # Pisahkan data menjadi train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Inisialisasi hasil
        results = []

        # 1. Decision Tree
        dt_model = DecisionTreeClassifier(random_state=42)
        dt_model.fit(X_train, y_train)
        dt_pred = dt_model.predict(X_test)
        dt_report = classification_report(y_test, dt_pred, output_dict=True)
        dt_cm = confusion_matrix(y_test, dt_pred)
        results.append({
            "model_name": "Decision Tree",
            "report": dt_report,
            "confusion_matrix": dt_cm
        })

        # 2. SVM
        svm_model = SVC(kernel='rbf', C=1, gamma='scale', random_state=42)
        svm_model.fit(X_train, y_train)
        svm_pred = svm_model.predict(X_test)
        svm_report = classification_report(y_test, svm_pred, output_dict=True)
        svm_cm = confusion_matrix(y_test, svm_pred)
        results.append({
            "model_name": "Support Vector Machine (SVM)",
            "report": svm_report,
            "confusion_matrix": svm_cm
        })

        # Membuat plot untuk setiap model
        plot_urls = []
        for result in results:
            plt.figure(figsize=(6, 4))
            sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues', cbar=False)
            plt.title(f"Confusion Matrix ({result['model_name']})")
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            img = io.BytesIO()
            plt.savefig(img, format='png')
            img.seek(0)
            plot_url = base64.b64encode(img.getvalue()).decode()
            plt.close()
            plot_urls.append(plot_url)

        # Kirimkan hasil evaluasi ke template
        return render_template(
            'classification.html',
            title="Classification Results",
            results=results,
            plot_urls=plot_urls,
            active_page="classification"
        )

    except Exception as e:
        return f"ERROR: {str(e)}", 500


@app.route('/association')
def association():
    # Load data
    data = pd.read_csv("user_behavior_dataset.csv")

    # Preprocessing
    data = data[['Device Model', 'Operating System', 'App Usage Time (min/day)', 'Number of Apps Installed', 'Screen On Time (hours/day)', 'Battery Drain (mAh/day)', 'Data Usage (MB/day)', 'Age', 'Gender']]
    grouped_data = data.groupby(['Device Model', 'Operating System']).agg({
        'App Usage Time (min/day)': 'mean',
        'Number of Apps Installed': 'mean'
    }).reset_index()

    grouped_data['App Usage Category'] = pd.cut(grouped_data['App Usage Time (min/day)'],
                                                bins=[0, 1, 2, 3, 4, 5, 24],
                                                labels=['<1h', '1-2h', '2-3h', '3-4h', '4-5h', '>5h'])

    association_data = grouped_data[['Device Model', 'Operating System', 'App Usage Category', 'Number of Apps Installed']]

    transaction_list = []
    for _, row in association_data.iterrows():
        transaction = [
            row['Device Model'],
            row['Operating System'],
            row['App Usage Category'],
            f"Apps_{row['Number of Apps Installed']}"
        ]
        transaction_list.append(transaction)

    transaction_df = pd.DataFrame(transaction_list)
    onehot = pd.get_dummies(transaction_df.stack()).groupby(level=0).sum()

    # FP-Growth
    frequent_itemsets_fp = fpgrowth(onehot, min_support=0.05, use_colnames=True)
    rules_fp = association_rules(frequent_itemsets_fp, metric="confidence", min_threshold=0.5, num_itemsets=len(frequent_itemsets_fp))

    # Apriori
    frequent_itemsets_apriori = apriori(onehot, min_support=0.05, use_colnames=True)
    rules_apriori = association_rules(frequent_itemsets_apriori, metric="confidence", min_threshold=0.5, num_itemsets=len(frequent_itemsets_apriori))

    # Visualization for FP-Growth
    heatmap_img_fp = io.BytesIO()
    heatmap_data_fp = rules_fp.pivot_table(index='antecedents', columns='consequents', values='confidence', fill_value=0)
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data_fp, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Heatmap of FP-Growth Rules')
    plt.savefig(heatmap_img_fp, format='png')
    heatmap_img_fp.seek(0)
    heatmap_url_fp = base64.b64encode(heatmap_img_fp.getvalue()).decode('utf8')

    # Visualization for Apriori
    heatmap_img_apriori = io.BytesIO()
    heatmap_data_apriori = rules_apriori.pivot_table(index='antecedents', columns='consequents', values='confidence', fill_value=0)
    plt.figure(figsize=(12, 8))
    sns.heatmap(heatmap_data_apriori, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Heatmap of Apriori Rules')
    plt.savefig(heatmap_img_apriori, format='png')
    heatmap_img_apriori.seek(0)
    heatmap_url_apriori = base64.b64encode(heatmap_img_apriori.getvalue()).decode('utf8')

    # Graph Visualization FP
    graph_img_fp = io.BytesIO()
    G_fp = nx.DiGraph()
    for _, row in rules_fp.iterrows():
        for antecedent in row['antecedents']:
            G_fp.add_edge(antecedent, row['consequents'], weight=row['confidence'])
    pos_fp = nx.spring_layout(G_fp)
    plt.figure(figsize=(12, 12))
    nx.draw(G_fp, pos_fp, with_labels=True, node_size=2000, node_color='lightblue', font_size=10, font_weight='bold', arrows=True)
    edge_labels_fp = nx.get_edge_attributes(G_fp, 'weight')
    nx.draw_networkx_edge_labels(G_fp, pos_fp, edge_labels=edge_labels_fp)
    plt.savefig(graph_img_fp, format='png')
    graph_img_fp.seek(0)
    graph_url_fp = base64.b64encode(graph_img_fp.getvalue()).decode('utf8')

    # Graph Visualization Apriori 
    graph_img_apriori = io.BytesIO()
    G_apriori = nx.DiGraph()
    for _, row in rules_apriori.iterrows():
        for antecedent in row['antecedents']:
            G_apriori.add_edge(antecedent, row['consequents'], weight=row['confidence'])
    pos_apriori = nx.spring_layout(G_apriori)
    plt.figure(figsize=(12, 12))
    nx.draw(G_apriori, pos_apriori, with_labels=True, node_size=2000, node_color='lightblue', font_size=10, font_weight='bold', arrows=True)
    edge_labels_fp = nx.get_edge_attributes(G_apriori, 'weight')
    nx.draw_networkx_edge_labels(G_apriori, pos_apriori, edge_labels=edge_labels_fp)
    plt.savefig(graph_img_apriori, format='png')
    graph_img_apriori.seek(0)
    graph_url_apriori = base64.b64encode(graph_img_apriori.getvalue()).decode('utf8')


    # Generate Device Model
    device_model_counts = data['Device Model'].value_counts().head(10)
    operating_system_counts = data['Operating System'].value_counts().head(10)
    device_img = io.BytesIO()
    plt.figure(figsize=(12, 6))
    sns.barplot(x=device_model_counts.values, y=device_model_counts.index, palette='viridis')
    plt.title('10 Device Model Teratas')
    plt.xlabel('Jumlah Penggunaan')
    plt.ylabel('Device Model')
    plt.savefig(device_img, format='png')
    device_img.seek(0)
    device_url = base64.b64encode(device_img.getvalue()).decode('utf8')

    # Generate OS
    os_img = io.BytesIO()
    plt.figure(figsize=(12, 6))
    sns.barplot(x=operating_system_counts.values, y=operating_system_counts.index, palette='viridis')
    plt.title('10 Operating System Teratas')
    plt.xlabel('Jumlah Penggunaan')
    plt.ylabel('Operating System')
    plt.savefig(os_img, format='png')
    os_img.seek(0)
    os_url = base64.b64encode(os_img.getvalue()).decode('utf8')

    # Generate dvm 0
    dvm = data.groupby('Device Model').agg({
    'Number of Apps Installed': 'mean',
    'App Usage Time (min/day)': 'mean'
    }).reset_index()
    print(dvm)
    plt.figure(figsize=(12, 6))
    dvm_img = io.BytesIO()
    sns.scatterplot(data=dvm, x='Number of Apps Installed', y='App Usage Time (min/day)', hue='Device Model', palette='rainbow', s=100)
    plt.title('Hubungan antara Number of Apps Installed dan App Usage Time')
    plt.xlabel('Rata-rata Jumlah Aplikasi yang Diinstal')
    plt.ylabel('Rata-rata Waktu Penggunaan Aplikasi (dalam menit)')
    plt.legend(title="Device Model", fontsize=10)
    plt.savefig(dvm_img, format='png')
    dvm_img.seek(0)
    dvm_url = base64.b64encode(dvm_img.getvalue()).decode('utf8')

    # Generate dvm1
    dvm1 = data.groupby('Device Model').agg({
        'App Usage Time (min/day)': 'mean',
        'Screen On Time (hours/day)': 'mean',
        'Battery Drain (mAh/day)': 'mean'
     }).reset_index()
    print(dvm1)
    plt.figure(figsize=(12, 6))
    dvm1_img = io.BytesIO()
    sns.scatterplot(data=dvm1, x='App Usage Time (min/day)', y='Screen On Time (hours/day)', hue='Device Model', palette='rainbow', s=100)
    plt.title('Hubungan antara App Usage Time dan Screen On Time')
    plt.xlabel('Rata-rata Waktu Penggunaan Aplikasi (dalam menit)')
    plt.ylabel('Rata-rata Waktu Layar Menyala (dalam menit)')
    plt.legend(title="Device Model", fontsize=10)
    plt.savefig(dvm1_img, format='png')
    dvm1_img.seek(0)
    dvm1_url = base64.b64encode(dvm1_img.getvalue()).decode('utf8')


    # Generate dvm2
    correlation_usage_screen = dvm1['App Usage Time (min/day)'].corr(dvm1['Screen On Time (hours/day)'])
    print(f'Korelasi antara App Usage Time dan Screen On Time: {correlation_usage_screen}')

    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=dvm1, x='Screen On Time (hours/day)', y='Battery Drain (mAh/day)', hue='Device Model', palette='rainbow', s=100)
    plt.title('Hubungan antara Screen On Time dan Battery Drain')
    plt.xlabel('Rata-rata Waktu Layar Menyala (dalam menit)')
    plt.ylabel('Rata-rata Pengurasan Baterai (dalam %)')
    plt.legend(title="Device Model", fontsize=10)
        
    dvm2_img = io.BytesIO()
    plt.savefig(dvm2_img, format='png')
    dvm2_img.seek(0)
    dvm2_url = base64.b64encode(dvm2_img.getvalue()).decode('utf8')

    correlation_screen_battery = dvm1['Screen On Time (hours/day)'].corr(dvm1['Battery Drain (mAh/day)'])
    print(f'Korelasi antara Screen On Time dan Battery Drain: {correlation_screen_battery}')


    # Generate dvm 3
    device_model_analysis = data.groupby('Device Model').agg({
        'Screen On Time (hours/day)': 'mean',
        'Data Usage (MB/day)': 'mean'
    }).reset_index()

    print(device_model_analysis)

    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=device_model_analysis, x='Screen On Time (hours/day)', y='Data Usage (MB/day)', hue='Device Model', palette='rainbow', s=100)
    plt.title('Hubungan antara Screen On Time dan Data Usage')
    plt.xlabel('Rata-rata Waktu Layar Menyala (dalam menit)')
    plt.ylabel('Rata-rata Penggunaan Data (dalam MB)')
    plt.legend(title="Device Model", fontsize=10)

    dvm3_img = io.BytesIO()
    plt.savefig(dvm3_img, format='png')
    dvm3_img.seek(0)
    dvm3_url = base64.b64encode(dvm3_img.getvalue()).decode('utf8')


    # Generate Gender
    gender_img = io.BytesIO()
    plt.figure(figsize=(25, 10))
    gender_analysis = data.groupby('Gender')['Screen On Time (hours/day)'].mean().reset_index()
    print(gender_analysis)
    plt.figure(figsize=(8, 5))
    sns.barplot(data=gender_analysis, x='Gender', y='Screen On Time (hours/day)', palette='rainbow')
    plt.title('Rata-rata Screen On Time berdasarkan Gender')
    plt.xlabel('Gender')
    plt.ylabel('Rata-rata Waktu Layar Menyala (dalam menit)')
    plt.savefig(gender_img, format='png')
    gender_img.seek(0)
    gender_url = base64.b64encode(gender_img.getvalue()).decode('utf8')


  # Generate Age
    age_img = io.BytesIO()  
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=data, x='Age', y='Screen On Time (hours/day)', palette='rainbow')
    plt.title('Distribusi Screen On Time berdasarkan Usia')
    plt.xlabel('Usia')
    plt.ylabel('Waktu Layar Menyala (dalam menit)')
    plt.xticks(rotation=45)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(age_img, format='png')
    age_img.seek(0)
    age_url = base64.b64encode(age_img.getvalue()).decode('utf8')


    # Generate Scatter Plot
    scatter_img = io.BytesIO()
    plt.figure(figsize=(12, 6))
    sns.scatterplot(data=grouped_data, x='Number of Apps Installed', y='App Usage Time (min/day)', hue='Device Model', palette='rainbow', s=100)
    plt.title('Number of Apps Installed vs App Usage Time')
    plt.xlabel('Number of Apps Installed')
    plt.ylabel('App Usage Time (min/day)')
    plt.legend(title="Device Model", fontsize=10)
    plt.savefig(scatter_img, format='png')
    scatter_img.seek(0)
    scatter_url = base64.b64encode(scatter_img.getvalue()).decode('utf8')




    return render_template('association.html',
                           title="Association Analysis",
                           header="FP-Growth and Apriori Rules",
                           heatmap_url_fp=heatmap_url_fp,
                           heatmap_url_apriori=heatmap_url_apriori,
                           graph_url_fp=graph_url_fp,
                           graph_url_apriori = graph_url_apriori,
                           scatter_url=scatter_url,
                           device_url = device_url,
                           os_url=os_url,
                           dvm_url = dvm_url,
                           dvm1_url = dvm1_url,
                           dvm2_url = dvm2_url,
                           dvm3_url = dvm3_url,
                           gender_url = gender_url,
                           age_url = age_url,
                           active_page="association")


@app.route('/prediksi', methods=['GET', 'POST'])
def prediksi():
    if request.method == 'POST':
        try:
            # Ambil input dari form
            app_usage_time = float(request.form['app_usage_time'])
            screen_on_time = float(request.form['screen_on_time'])
            data_usage = float(request.form['data_usage'])
            num_apps_installed = int(request.form['num_apps_installed'])
            age = int(request.form['age'])
            algorithm = request.form['algorithm']  # Algoritma yang dipilih oleh pengguna

            # Preprocess input
            input_data = pd.DataFrame({
                'App Usage Time (min/day)': [app_usage_time],
                'Screen On Time (hours/day)': [screen_on_time],
                'Data Usage (MB/day)': [data_usage],
                'Number of Apps Installed': [num_apps_installed],
                'Age': [age]
            })

            # Load scaler dan model
            scaler = joblib.load('scaler.pkl')
            if algorithm == 'random_forest':
                model = joblib.load('rf_model.pkl')
            elif algorithm == 'linear_regression':
                model = joblib.load('lr_model.pkl')
            else:
                return "Invalid algorithm selected.", 400

            # Scale input data
            input_data_scaled = scaler.transform(input_data)

            # Prediksi konsumsi baterai
            prediction = model.predict(input_data_scaled)
            first_prediction = round(prediction[0], 2)

            # Tampilkan hasil prediksi
            return render_template(
                'prediksi.html',
                title="Prediksi Konsumsi Baterai",
                header="Prediksi Konsumsi Baterai",
                prediction=first_prediction,
                algorithm=algorithm,
                active_page="prediksi"
            )
        except Exception as e:
            return f"Error occurred: {str(e)}", 400
    return render_template('prediksi.html',
                           title="Prediksi Konsumsi Baterai",
                           header="Prediksi Konsumsi Baterai",
                           active_page="prediksi")


@app.route("/deteksi", methods=["GET", "POST"])
def deteksi():
    features = [
        "App Usage Time (min/day)",
        "Screen On Time (hours/day)",
        "Battery Drain (mAh/day)",
        "Number of Apps Installed",
        "Data Usage (MB/day)",
    ]
    if request.method == "POST":
        # Normalisasi data
        X = df[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Model Isolation Forest
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        anomaly_labels = iso_forest.fit_predict(X_scaled)

        # Tambahkan label anomaly ke dataset
        df["anomaly"] = anomaly_labels

        # Statistik untuk range Non-Anomali
        non_anomalies = df[df["anomaly"] == 1]
        stats = non_anomalies[features].agg(["mean", "std", "min", "max"]).T
        deviation = 1.4
        stats["normal_range_min"] = stats["mean"] - deviation * stats["std"]
        stats["normal_range_max"] = stats["mean"] + deviation * stats["std"]

        for feature in features:
         if stats.loc[feature, "min"] >= 0:  # Jika negatif, maka akan diset ke 0
            stats.loc[feature, "normal_range_min"] = max(0, stats.loc[feature, "normal_range_min"])


        for idx, row in df.iterrows():
            if row["anomaly"] == -1:  # Hanya untuk anomali
                log_details = []
                for feature in features:
                    normal_min = stats.loc[feature, "normal_range_min"]
                    normal_max = stats.loc[feature, "normal_range_max"]
                    if not (normal_min <= row[feature] <= normal_max):
                        log_details.append(f"{feature} ({row[feature]:.2f})")
                if log_details:
                    df.at[idx, "Log"] = f"Anomali karena: {', '.join(log_details)}"
                else:
                    df.at[idx, "Log"] = "Anomali berdasarkan kombinasi fitur."

        # Filter data anomali
        anomalies = df[df["anomaly"] == -1]

        return render_template(
            "deteksi.html",
            title="Deteksi Anomali",
            header="Hasil Deteksi Anomali",
            anomalies=anomalies.to_dict(orient="records"),
            columns=df.columns,
            stats=stats.to_dict(orient="index"),  # Kirim statistik
        )

    return render_template(
        "deteksi.html",
        title="Deteksi Anomali",
        header="Deteksi Perilaku Pengguna Tidak Normal",
    )

if __name__ == '__main__':
    app.run(debug=True)
