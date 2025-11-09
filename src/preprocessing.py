import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

def load_and_preprocess(path='./data/depression_anxiety.csv', k_best=6, pca_components=4):
    # Load dataset
    df = pd.read_csv(path)
    print("Available columns:", df.columns.tolist())

    # Select relevant columns
    selected_columns = [
        'age', 'gender', 'bmi', 'who_bmi',
        'phq_score', 'epworth_score', 'sleepiness',
        'depression_severity'
    ]
    df = df[selected_columns].dropna()

    # Encode categorical columns
    label_enc = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = label_enc.fit_transform(df[col])

    # Remove outliers using IQR
    for col in df.select_dtypes(include=np.number).columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]

    # Separate features and target
    X = df.drop('depression_severity', axis=1)
    y = label_enc.fit_transform(df['depression_severity'])

    # Remove constant features
    vt = VarianceThreshold(threshold=0.0)
    X_nonconstant = vt.fit_transform(X)

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_nonconstant)

    # Feature selection
    X_kbest = SelectKBest(score_func=f_classif, k=k_best).fit_transform(X_scaled, y)

    # PCA for quantum kernel efficiency
    pca = PCA(n_components=pca_components)
    X_pca = pca.fit_transform(X_kbest)

    # Balance classes
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X_pca, y)

    return X_resampled, y_resampled

# Run directly to test
if __name__ == "__main__":
    X, y = load_and_preprocess()
    print("Enhanced preprocessing complete.")
    print("Feature shape:", X.shape)
    print("Target shape:", y.shape)