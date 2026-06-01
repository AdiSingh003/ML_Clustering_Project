import pandas as pd
import numpy as np
import time
import os
import sqlite3
import json
import pickle
import joblib
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Absolute import if needed, but relative or module import is better. 
# We'll use the DB constants from our db_manager.
from backend.db_manager import DB_PATH, JOBLIB_PATH

def load_data():
    file_id = '1Eld6cmwQVh_Ilobnne0PJ12zXnIwJGZD'
    url = f'https://drive.google.com/uc?id={file_id}'
    df = pd.read_csv(url)
    return df

def preprocess_data(df):
    df_new = df.drop(['ip.address','full.name','index'], axis=1)
    df_new['amount_per_item'] = df['amount'] / (df['items'] + 1e-5)
    df_new = df_new.drop(['items'], axis=1)
    return df_new

def split_data(df_new):
    X_temp, X_test = train_test_split(df_new, test_size=0.2, random_state=42)
    X_train, X_val = train_test_split(X_temp, test_size=0.25, random_state=42)
    return X_train, X_val, X_test

def engineer_features(X_train, X_val, X_test):
    cols_to_encode = ['region']
    cols_to_scale = ["age","amount","amount_per_item"]
    cols_passthrough = ['in.store']

    preprocessor = ColumnTransformer([
        ('cat', OneHotEncoder(sparse_output=False), cols_to_encode),
        ('pass', 'passthrough', cols_passthrough),
        ('num', MinMaxScaler(), cols_to_scale),
        ], remainder='drop')

    X_train_scaled = preprocessor.fit_transform(X_train)
    X_val_scaled = preprocessor.transform(X_val)
    X_test_scaled = preprocessor.transform(X_test)
    
    ohe_feature_names = list(preprocessor.named_transformers_['cat'].get_feature_names_out(cols_to_encode))
    all_feature_names = ohe_feature_names + cols_passthrough + cols_to_scale
    
    return X_train_scaled, X_val_scaled, X_test_scaled, all_feature_names, preprocessor


def tune_kmeans(X_train_scaled):
    conn = sqlite3.connect(DB_PATH)
    
    # Check if we already have the results
    df_existing = pd.read_sql_query("SELECT * FROM kfold_tuning ORDER BY k", conn)
    k_values = list(range(2, 11))
    
    if not df_existing.empty and len(df_existing) == len(k_values):
        best_k = int(df_existing.loc[df_existing['avg_silhouette'].idxmax()]['k'])
        silhouette_avgs = df_existing['avg_silhouette'].tolist()
        fold_scores = []
        if 'fold_scores' in df_existing.columns:
            fold_scores = [json.loads(item) if isinstance(item, str) else [] for item in df_existing['fold_scores'].tolist()]
        df_existing = df_existing.rename(columns={
            "k": "k",
            "avg_silhouette": "Avg Silhouette",
            "avg_iters": "Avg Iters",
            "time_taken": "Time (s)"
        })
        conn.close()
        # Return order must match the non-cached flow: (df, k_values, silhouette_avgs, fold_scores, best_k, loaded_flag)
        return df_existing, k_values, silhouette_avgs, fold_scores, best_k, True

    # Otherwise, compute and store
    best_k = None
    best_score = -1
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    silhouette_avgs = []
    results = []
    fold_scores_all = []

    cursor = conn.cursor()
    cursor.execute("DELETE FROM kfold_tuning")

    for k in k_values:
        fold_scores = []
        fold_iters = []
        start = time.time()

        for train_idx, val_idx in kf.split(X_train_scaled):
            X_fold_train = X_train_scaled[train_idx]
            model = KMeans(n_clusters=k, init='k-means++', n_init=10, max_iter=100, random_state=42)
            labels = model.fit_predict(X_fold_train)
            score = silhouette_score(X_fold_train, labels)
            fold_scores.append(score)
            fold_iters.append(model.n_iter_)

        # collect per-k fold scores
        fold_scores_all.append(fold_scores)

        avg_score = float(np.mean(fold_scores))
        silhouette_avgs.append(avg_score)
        avg_iters = float(np.mean(fold_iters))
        time_taken = time.time() - start

        # Save to DB immediately (store fold scores for this k)
        cursor.execute('''
            INSERT INTO kfold_tuning (k, avg_silhouette, avg_iters, time_taken, fold_scores)
            VALUES (?, ?, ?, ?, ?)
        ''', (k, float(avg_score), float(avg_iters), float(time_taken), json.dumps(fold_scores)))

        results.append({
            "k": k,
            "Avg Silhouette": round(avg_score, 3),
            "Avg Iters": round(avg_iters, 1),
            "Time (s)": round(time_taken, 2)
        })

        if avg_score > best_score:
            best_score = avg_score
            best_k = k

    conn.commit()
    conn.close()
    # fold_scores_all is a list of lists (one entry per k); keep return order consistent
    return pd.DataFrame(results), k_values, silhouette_avgs, fold_scores_all, best_k, False


def final_train_and_pca(X_train_scaled, X_val_scaled, X_test_scaled, best_k, feature_names):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('SELECT cached_data FROM final_training_results WHERE best_k = ?', (best_k,))
    row = cursor.fetchone()
    if row is not None and row[0] is not None:
        try:
            cached_data = pickle.loads(row[0])
            conn.close()
            return (
                cached_data['train_score'],
                cached_data['val_score'],
                cached_data['test_score'],
                cached_data['train_pca_df'],
                cached_data['val_pca_df'],
                cached_data['test_pca_df'],
                cached_data['variance'],
                cached_data['pca_loadings'],
                cached_data['top_features_df'],
                cached_data['cluster_summary_test'],
                True
            )
        except Exception:
            pass

    X_train_val_scaled = np.vstack([X_train_scaled, X_val_scaled])
    final_kmeans = KMeans(n_clusters=best_k, init='k-means++', n_init=10, max_iter=100, random_state=42)
    final_kmeans.fit(X_train_val_scaled)

    train_labels = final_kmeans.predict(X_train_scaled)
    val_labels = final_kmeans.predict(X_val_scaled)
    test_labels = final_kmeans.predict(X_test_scaled)

    train_score = silhouette_score(X_train_scaled, train_labels)
    val_score = silhouette_score(X_val_scaled, val_labels)
    test_score = silhouette_score(X_test_scaled, test_labels)

    pca = PCA(n_components=3, random_state=42)
    pca.fit(X_train_val_scaled)

    train_pca = pca.transform(X_train_scaled)
    val_pca = pca.transform(X_val_scaled)
    test_pca = pca.transform(X_test_scaled)

    train_pca_df = pd.DataFrame(train_pca, columns=['PC1', 'PC2', 'PC3'])
    train_pca_df['Cluster'] = train_labels.astype(str)

    val_pca_df = pd.DataFrame(val_pca, columns=['PC1', 'PC2', 'PC3'])
    val_pca_df['Cluster'] = val_labels.astype(str)

    test_pca_df = pd.DataFrame(test_pca, columns=['PC1', 'PC2', 'PC3'])
    test_pca_df['Cluster'] = test_labels.astype(str)

    variance = np.sum(pca.explained_variance_ratio_) * 100

    pca_loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca.n_components_)],
        index=feature_names
    )

    top_features = {}
    for i in range(pca.n_components_):
        pc_name = f"PC{i+1}"
        top_features[pc_name] = pca_loadings.iloc[:, i].abs().sort_values(ascending=False).head(5).index.tolist()
    top_features_df = pd.DataFrame(top_features)

    X_test_df = pd.DataFrame(X_test_scaled, columns=feature_names)
    X_test_df['Cluster'] = test_labels
    cluster_summary_test = X_test_df.groupby('Cluster').mean()

    cached_data = {
        'train_score': train_score,
        'val_score': val_score,
        'test_score': test_score,
        'train_pca_df': train_pca_df,
        'val_pca_df': val_pca_df,
        'test_pca_df': test_pca_df,
        'variance': variance,
        'pca_loadings': pca_loadings,
        'top_features_df': top_features_df,
        'cluster_summary_test': cluster_summary_test
    }
    serialized = pickle.dumps(cached_data)
    cursor.execute('''
        INSERT OR REPLACE INTO final_training_results 
        (best_k, train_score, val_score, test_score, variance, cached_data, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        best_k,
        float(train_score),
        float(val_score),
        float(test_score),
        float(variance),
        sqlite3.Binary(serialized),
        time.strftime('%Y-%m-%d %H:%M:%S')
    ))
    conn.commit()
    conn.close()

    return (
        train_score,
        val_score,
        test_score,
        train_pca_df,
        val_pca_df,
        test_pca_df,
        variance,
        pca_loadings,
        top_features_df,
        cluster_summary_test,
        False
    )
