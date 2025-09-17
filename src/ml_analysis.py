import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

def cluster_trades(df: pd.DataFrame, n_clusters=3):
    features = df[['Quantity', 'Rate', 'Amount']].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    df['Cluster'] = kmeans.fit_predict(X_scaled)
    return df, kmeans

def classify_trade_size(df: pd.DataFrame):
    # Assume binary classification Large=1 if Quantity > median, else Small=0
    df = df.copy()
    df['TradeSizeBinary'] = (df['Quantity'] > df['Quantity'].median()).astype(int)
    
    features = df[['Rate', 'Amount']]
    target = df['TradeSizeBinary']
    
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=42)
    
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    
    df.loc[X_test.index, 'PredictedTradeSize'] = y_pred
    return df, clf

if __name__ == '__main__':
    enhanced_path = "../data/processed/enhanced_dataset.csv"
    df = pd.read_csv(enhanced_path)
    
    # Clustering
    df_clustered, kmeans_model = cluster_trades(df)
    print("Clusters assigned:\n", df_clustered['Cluster'].value_counts())
    
    # Classification
    df_classified, clf = classify_trade_size(df_clustered)
