import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.metrics import silhouette_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from scipy.stats import median_abs_deviation
import nltk
from nltk.tokenize import word_tokenize
import os
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances
from tensorflow.keras.layers import Dropout

# Download NLTK resources
nltk.download('punkt')

def process_childes_transcript(file_path):
    """
    Process a CHILDES transcript file to extract therapy-related features.
    Returns: dict with interaction_frequency, engagement_score, utterance_length.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            text = file.read()
        
        lines = text.split('\n')
        child_utts = [line.replace('*CHI:', '').strip() for line in lines if line.startswith('*CHI')]
        adult_utts = [line.replace('*ADU:', '').strip() for line in lines if line.startswith('*ADU')]
        
        interaction_frequency = len(child_utts) + len(adult_utts)
        total_utts = len(child_utts) + len(adult_utts)
        engagement_score = len(child_utts) / total_utts if total_utts > 0 else 0
        child_words = [len(word_tokenize(utt)) for utt in child_utts if utt]
        utterance_length = sum(child_words) / len(child_words) if child_words else 0
        
        return {
            'interaction_frequency': interaction_frequency,
            'engagement_score': engagement_score,
            'utterance_length': utterance_length
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def process_childes_directory(directory):
    """
    Process all .txt files in a CHILDES directory.
    Returns: DataFrame with extracted features.
    """
    data = []
    for filename in os.listdir(directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(directory, filename)
            features = process_childes_transcript(file_path)
            if features:
                features['file'] = filename
                data.append(features)
    
    return pd.DataFrame(data)

def generate_synthetic_data(n_ndar=1000, n_childes=500):
    """
    Generate synthetic NDAR and CHILDES data mimicking real therapy data.
    Returns: Combined DataFrame with source indicator.
    """
    np.random.seed(42)
    # Mixture of Gaussians for NDAR
    ndar_clusters = 3
    ndar_data = {k: [] for k in ['interaction_frequency', 'session_duration', 'engagement_score', 'repetitive_behavior', 'utterance_length', 'source']}
    for i in range(ndar_clusters):
        size = n_ndar // ndar_clusters
        ndar_data['interaction_frequency'] += list(np.random.normal(12 + i*2, 2, size))
        ndar_data['session_duration'] += list(np.random.normal(50 + i*5, 10, size))
        ndar_data['engagement_score'] += list(np.random.normal(6 + i, 1, size))
        ndar_data['repetitive_behavior'] += list(np.random.normal(5 + i, 1, size))
        ndar_data['utterance_length'] += list(np.zeros(size))
        ndar_data['source'] += ['NDAR'] * size
    # Mixture of Gaussians for CHILDES
    childes_clusters = 2
    childes_data = {k: [] for k in ['interaction_frequency', 'session_duration', 'engagement_score', 'repetitive_behavior', 'utterance_length', 'source']}
    for i in range(childes_clusters):
        size = n_childes // childes_clusters
        childes_data['interaction_frequency'] += list(np.random.normal(10 + i*1.5, 1, size))
        childes_data['session_duration'] += list(np.random.normal(45 + i*3, 5, size))
        childes_data['engagement_score'] += list(np.random.normal(0.6 + i*0.2, 0.05, size))
        childes_data['repetitive_behavior'] += list(np.random.normal(4 + i*0.5, 0.5, size))
        childes_data['utterance_length'] += list(np.random.normal(3 + i*0.5, 0.5, size))
        childes_data['source'] += ['CHILDES'] * size
    df_ndar = pd.DataFrame(ndar_data)
    df_childes = pd.DataFrame(childes_data)
    return pd.concat([df_ndar, df_childes], ignore_index=True)

def load_childes_features(path):
    df = pd.read_csv(path)
    # Avoid division by zero for engagement_score
    df['engagement_score'] = df['child_TNW'] / (df['child_TNW'] + df['examiner_TNW']).replace(0, np.nan)
    return pd.DataFrame({
        'interaction_frequency': df['child_TNW'],
        'session_duration': df['age'],  # or df['age_years']
        'engagement_score': df['engagement_score'],
        'repetitive_behavior': df['repetition'],
        'utterance_length': df['mlu_words'],
        'source': 'CHILDES'
    })

def load_ndar_features(path):
    df = pd.read_csv(path)
    a_cols = [f'A{i}' for i in range(1, 11)]
    df['A_sum'] = df[a_cols].sum(axis=1)
    df['repetitive_behavior'] = df[['A2', 'A5', 'A7']].sum(axis=1)
    return pd.DataFrame({
        'interaction_frequency': df['A_sum'],
        'session_duration': df['Age'],
        'engagement_score': 10 - df['A_sum'],
        'repetitive_behavior': df['repetitive_behavior'],
        'utterance_length': 0,
        'source': 'NDAR'
    })

def load_combined_data(childes_path, ndar_path):
    df_childes = load_childes_features(childes_path)
    df_ndar = load_ndar_features(ndar_path)
    return pd.concat([df_childes, df_ndar], ignore_index=True)

def preprocess_data(df):
    """
    Preprocess data: impute missing values, filter outliers, normalize, apply PCA.
    Returns: Preprocessed DataFrame, scaled DataFrame, reduced DataFrame.
    """
    # Preserve source column
    source = df['source'] if 'source' in df.columns else None
    features = [col for col in df.columns if col != 'source']

    # Impute missing values
    imputer = KNNImputer(n_neighbors=5)
    df_features = pd.DataFrame(imputer.fit_transform(df[features]), columns=features)

    # Normalize engagement_score separately for CHILDES and NDAR
    if 'engagement_score' in df_features.columns and source is not None:
        for group in ['CHILDES', 'NDAR']:
            idx = source[source == group].index
            if not idx.empty:
                min_val = df_features.loc[idx, 'engagement_score'].min()
                max_val = df_features.loc[idx, 'engagement_score'].max()
                if max_val > min_val:
                    df_features.loc[idx, 'engagement_score'] = (
                        (df_features.loc[idx, 'engagement_score'] - min_val) /
                        (max_val - min_val)
                    )
                else:
                    df_features.loc[idx, 'engagement_score'] = 0.0

    # Filter outliers using MAD
    mad = median_abs_deviation(df_features, axis=0)
    median = np.median(df_features, axis=0)
    # Calculate outlier mask for each column
    outlier_mask = ((df_features - median).abs() / mad) < 3.5
    # Keep rows that are not outliers in at least one column
    df_filtered = df_features[outlier_mask.any(axis=1)]
    # Only update df_features if filtering does not remove all data
    if not df_filtered.empty:
        df_features = df_filtered

    # Update source if present
    if source is not None:
        df_features['source'] = source.loc[df_features.index]

    # Normalize all features (except source)
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df_features[features]), columns=features, index=df_features.index)
    if source is not None:
        df_scaled['source'] = df_features['source']

    # PCA
    pca = PCA(n_components=0.95)
    df_reduced = pd.DataFrame(pca.fit_transform(df_scaled[features]), columns=[f'PC{i+1}' for i in range(pca.n_components_)], index=df_scaled.index)

    return df_features, df_scaled, df_reduced

def apply_unsupervised_learning(df_scaled, df_reduced):
    """
    Apply K-means, DBSCAN, and Autoencoders with grid search for DBSCAN.
    Returns: DataFrame with cluster labels, anomaly scores, and silhouette scores.
    """
    df = df_scaled.copy()
    features = [col for col in df_scaled.columns if col != 'source']
    
    # K-means
    kmeans = KMeans(n_clusters=4, random_state=42)
    df['kmeans_cluster'] = kmeans.fit_predict(df_reduced)
    kmeans_silhouette = silhouette_score(df_reduced, df['kmeans_cluster'])
    
    # DBSCAN with much finer/smaller eps grid and lower min_samples
    best_dbscan = None
    best_silhouette = -1
    best_params = {}
    eps_values = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.12, 0.15, 0.18, 0.2]
    min_samples_values = [2, 3, 4, 5, 10]
    
    for eps in eps_values:
        for min_samples in min_samples_values:
            dbscan = DBSCAN(eps=eps, min_samples=min_samples)
            labels = dbscan.fit_predict(df_reduced)
            if len(set(labels)) > 1:  # Ensure multiple clusters
                score = silhouette_score(df_reduced, labels)
                if score > best_silhouette:
                    best_silhouette = score
                    best_dbscan = dbscan
                    best_params = {'eps': eps, 'min_samples': min_samples}
    
    if best_dbscan is None:
        print("DBSCAN grid search failed; using default parameters.")
        best_dbscan = DBSCAN(eps=0.3, min_samples=10)
        best_silhouette = -1
    
    df['dbscan_cluster'] = best_dbscan.fit_predict(df_reduced)
    dbscan_silhouette = best_silhouette
    
    # Autoencoder (deeper with dropout)
    autoencoder = Sequential([
        Dense(128, activation='relu', input_shape=(df_reduced.shape[1],)),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dense(df_reduced.shape[1], activation='sigmoid')
    ])
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(df_reduced, df_reduced, epochs=150, batch_size=32, verbose=0)
    
    reconstructions = autoencoder.predict(df_reduced, verbose=0)
    mse = np.mean(np.power(df_reduced - reconstructions, 2), axis=1)
    mse_scaled = (mse - np.min(mse)) / (np.max(mse) - np.min(mse)) if np.max(mse) != np.min(mse) else mse
    df['anomaly_score'] = mse_scaled
    df['anomaly'] = (mse_scaled > np.percentile(mse_scaled, 90)).astype(int)
    
    return df, kmeans_silhouette, dbscan_silhouette, best_params

def visualize_clusters(df, df_reduced):
    """
    Generate PCA plot for K-means clusters and DBSCAN clusters, and save PCA coordinates.
    Saves: cluster_plot.png, dbscan_cluster_plot.png, pca_coordinates.csv
    """
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(df_reduced['PC1'], df_reduced['PC2'], c=df['kmeans_cluster'], cmap='viridis')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('K-means Clusters in PCA Space')
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True)
    plt.savefig('cluster_plot.png')
    plt.close()
    # DBSCAN cluster plot
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(df_reduced['PC1'], df_reduced['PC2'], c=df['dbscan_cluster'], cmap='tab10')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('DBSCAN Clusters in PCA Space')
    plt.colorbar(scatter, label='DBSCAN Cluster')
    plt.grid(True)
    plt.savefig('dbscan_cluster_plot.png')
    plt.close()
    # Save PCA coordinates
    pca_df = df_reduced.copy()
    pca_df['kmeans_cluster'] = df['kmeans_cluster']
    pca_df['dbscan_cluster'] = df['dbscan_cluster']
    pca_df.to_csv('pca_coordinates.csv', index=False)

def main():
    """
    Main function to run the unsupervised learning framework.
    """
    # Load or generate data
    childes_path = 'data/childes_SLI.csv'
    ndar_path = 'data/Autism_Screening_Data_Combined.csv'
    if os.path.exists(childes_path) and os.path.exists(ndar_path):
        print("Loading real CHILDES and NDAR-aligned data...")
        df = load_combined_data(childes_path, ndar_path)
    else:
        print("Real data not found; using synthetic data.")
        df = generate_synthetic_data()
    
    # Preprocess data
    df, df_scaled, df_reduced = preprocess_data(df)
    
    # Distance matrix visualization for DBSCAN tuning
    dist_matrix = euclidean_distances(df_reduced)
    plt.hist(dist_matrix.flatten(), bins=50)
    plt.title('Distance Distribution for DBSCAN Tuning')
    plt.savefig('distance_distribution.png')
    plt.close()
    
    # Apply unsupervised learning
    df, kmeans_silhouette, dbscan_silhouette, dbscan_params = apply_unsupervised_learning(df_scaled, df_reduced)
    
    # Generate summaries
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    kmeans_summary = df.groupby('kmeans_cluster')[numeric_cols].mean()
    dbscan_summary = df[df['dbscan_cluster'] != -1].groupby('dbscan_cluster')[numeric_cols].mean() if len(set(df['dbscan_cluster'])) > 1 else pd.DataFrame()
    anomaly_summary = df[df['anomaly'] == 1][numeric_cols].mean()
    
    # Save results
    kmeans_summary.to_csv('kmeans_summary.csv')
    dbscan_summary.to_csv('dbscan_summary.csv') if not dbscan_summary.empty else print("No DBSCAN clusters to save.")
    pd.Series(anomaly_summary).to_csv('anomaly_summary.csv')
    
    # Save silhouette scores and DBSCAN parameters
    silhouette_scores = {
        'kmeans_silhouette': kmeans_silhouette,
        'dbscan_silhouette': dbscan_silhouette,
        'dbscan_eps': dbscan_params.get('eps', 0.3),
        'dbscan_min_samples': dbscan_params.get('min_samples', 10)
    }
    pd.Series(silhouette_scores).to_csv('silhouette_scores.csv')
    
    # Visualize
    visualize_clusters(df, df_reduced)
    
    # Print results
    print("K-means Cluster Summary:")
    print(kmeans_summary)
    print("\nDBSCAN Cluster Summary:")
    print(dbscan_summary)
    print("\nAnomaly Summary:")
    print(anomaly_summary)
    print("\nSilhouette Scores and DBSCAN Parameters:")
    print(f"K-means: {kmeans_silhouette:.3f}, DBSCAN: {dbscan_silhouette:.3f}, DBSCAN Params: {dbscan_params}")
    # Print CHILDES engagement score stats
    if 'source' in df.columns:
        print("CHILDES Engagement Scores:", df[df['source'] == 'CHILDES']['engagement_score'].describe())

if __name__ == "__main__":
    main()