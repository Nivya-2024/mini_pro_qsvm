import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE

from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit.circuit.library import PauliFeatureMap
from qiskit.utils import QuantumInstance
from qiskit import Aer
import time
import os

def phq_severity(score):
    if score <= 4: return "Minimal"
    elif score <= 9: return "Mild"
    elif score <= 14: return "Moderate"
    elif score <= 19: return "Moderately Severe"
    else: return "Severe"

def gad_severity(score):
    if score <= 4: return "Minimal"
    elif score <= 9: return "Mild"
    elif score <= 14: return "Moderate"
    else: return "Severe"

def run_qsvm_optimized():
    print("📥 Loading dataset...")
    df = pd.read_csv("data/depression_anxiety.csv")

    df["Depression_Level"] = df["phq_score"].apply(phq_severity)
    df["Anxiety_Level"] = df["gad_score"].apply(gad_severity)
    df["Risk_Index"] = df["phq_score"] + df["gad_score"] + df["epworth_score"] + (df["sleepiness"] / (df["epworth_score"] + 1))

    y = ((df["depression_diagnosis"] == 1) | (df["anxiety_diagnosis"] == 1)).astype(int)
    df["mental_risk_score"] = df["phq_score"] + df["gad_score"] + df["epworth_score"]
    df["sleep_ratio"] = df["sleepiness"] / (df["epworth_score"] + 1)
    df["phq_gad_interaction"] = df["phq_score"] * df["gad_score"]
    df["phq_squared"] = df["phq_score"] ** 2
    df["gad_epworth_ratio"] = df["gad_score"] / (df["epworth_score"] + 1)
    df["risk_index_interaction"] = df["Risk_Index"] * df["phq_score"]

    selected_features = [
        "phq_score", "gad_score", "epworth_score", "sleepiness", "Risk_Index",
        "mental_risk_score", "sleep_ratio", "phq_gad_interaction",
        "phq_squared", "gad_epworth_ratio", "risk_index_interaction"
    ]
    X = df[selected_features].copy()
    X.dropna(inplace=True)
    y = y.loc[X.index]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Apply PCA (4 components)
    pca = PCA(n_components=4)
    X_pca = pca.fit_transform(X_scaled)

    # 📤 Save PCA components and labels for dashboard
    pca_df = pd.DataFrame(X_pca[:, :2], columns=["PC1", "PC2"])
    pca_df["Diagnosis"] = y.values
    pca_df["PHQ"] = df.loc[X.index, "phq_score"].values
    pca_df["GAD"] = df.loc[X.index, "gad_score"].values
    pca_df["Epworth"] = df.loc[X.index, "epworth_score"].values
    os.makedirs("data", exist_ok=True)
    pca_df.to_csv("data/pca_dashboard.csv", index=False)

    # Apply SMOTE
    smote = SMOTE(sampling_strategy=0.8, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_pca, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
    )

    # Quantum training subset (300 samples)
    sss = StratifiedShuffleSplit(n_splits=1, train_size=300, random_state=42)
    for train_idx, _ in sss.split(X_train, y_train):
        qsvc_train_X = X_train[train_idx]
        qsvc_train_y = y_train.to_numpy()[train_idx]

    print("\n🧠 Training QSVM with PauliFeatureMap + PCA (4D)...")
    feature_map = PauliFeatureMap(
        feature_dimension=X_train.shape[1],
        reps=3,
        entanglement='full',
        paulis=['X', 'Y', 'Z']
    )
    backend = Aer.get_backend("aer_simulator_statevector")
    quantum_instance = QuantumInstance(backend=backend, shots=1024, seed_simulator=42, seed_transpiler=42)
    kernel = QuantumKernel(feature_map=feature_map, quantum_instance=quantum_instance)

    kernel_matrix = kernel.evaluate(qsvc_train_X)
    print("🔍 Kernel matrix sample:\n", kernel_matrix[:5, :5])

    try:
        qsvc = QSVC(quantum_kernel=kernel, C=1.0)
        start = time.time()
        qsvc.fit(qsvc_train_X, qsvc_train_y)
        duration = time.time() - start
        print(f"✅ QSVM trained in {round(duration, 2)} seconds")
        y_pred = qsvc.predict(X_test[:50])
    except Exception as e:
        print("❌ QSVM error:", e)
        y_pred = np.zeros(50)

    y_true = y_test[:50]
    acc = accuracy_score(y_true, y_pred)
    print(f"\n✅ Accuracy: {acc:.2f}")
    print("📊 Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("🧩 Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("🔍 ROC AUC Score:", roc_auc_score(y_true, y_pred))

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    print(f"🩺 Clinical Metrics:\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}")

if __name__ == "__main__":
    run_qsvm_optimized()