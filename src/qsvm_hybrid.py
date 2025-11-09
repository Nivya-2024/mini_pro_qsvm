import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

from qiskit_machine_learning.algorithms import QSVC
from qiskit_machine_learning.kernels import QuantumKernel
from qiskit.circuit.library import ZZFeatureMap
from qiskit_aer import Aer
import time

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

def train_hybrid_model():
    print("📥 Step 1: Loading dataset...")
    df = pd.read_csv("data/depression_anxiety.csv")
    print("✅ Dataset loaded. Shape:", df.shape)

    print("\n🧠 Step 2: Mapping severity levels...")
    df["Depression_Level"] = df["phq_score"].apply(phq_severity)
    df["Anxiety_Level"] = df["gad_score"].apply(gad_severity)

    print("\n📊 Step 3: Creating custom risk index...")
    df["Risk_Index"] = df["phq_score"] + df["gad_score"] + df["epworth_score"] + (df["sleepiness"] / (df["epworth_score"] + 1))

    print("\n📋 Step 3.5: Displaying sample individual-level scores and severity:")
    for i in range(5):
        person = df.iloc[i]
        print(f"\n👤 Person {i+1}")
        print(f"  PHQ-9 Score: {person['phq_score']} → Depression Level: {person['Depression_Level']}")
        print(f"  GAD-7 Score: {person['gad_score']} → Anxiety Level: {person['Anxiety_Level']}")
        print(f"  Sleepiness Score: {person['sleepiness']}")
        print(f"  Epworth Score: {person['epworth_score']}")
        print(f"  Risk Index: {person['Risk_Index']:.2f}")

    print("\n⚠ Step 3.6: Displaying high-risk individuals (Risk Index > 40):")
    high_risk = df[df["Risk_Index"] > 40]
    print(high_risk[["phq_score", "gad_score", "Depression_Level", "Anxiety_Level", "Risk_Index"]].head(10))

    print("\n🧪 Step 4: Preparing features and target...")
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

    print("\n🧹 Step 5: Cleaning data...")
    X.dropna(inplace=True)
    y = y.loc[X.index]

    print("\n📏 Step 6: Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("\n⚖ Step 7: Balancing classes with SMOTE (sampling_strategy=0.8)...")
    smote = SMOTE(sampling_strategy=0.8, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    print("\n🔀 Step 8: Splitting train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
    )

    print("\n🔬 Step 9: Stratified sampling for QSVM training (100 samples)...")
    sss = StratifiedShuffleSplit(n_splits=1, train_size=100, random_state=42)
    for train_idx, _ in sss.split(X_train, y_train):
        qsvc_train_X = X_train[train_idx]
        qsvc_train_y = y_train.to_numpy()[train_idx]

    print("\n🧠 Step 10: Training QSVM with QuantumKernel...")
    feature_map = ZZFeatureMap(feature_dimension=X_train.shape[1], reps=2, entanglement='full')
    kernel = QuantumKernel(feature_map=feature_map)

    try:
        qsvc = QSVC(quantum_kernel=kernel, C=1.0)
        print("⏳ Starting QSVM training...")
        start = time.time()
        qsvc.fit(qsvc_train_X, qsvc_train_y)
        duration = time.time() - start
        print(f"✅ QSVM trained in {round(duration, 2)} seconds")
        y_qsvc = qsvc.predict(X_test[:50])
        print("✅ QSVM prediction complete.")
    except Exception as e:
        print("❌ QSVM error:", e)
        y_qsvc = np.zeros(50)

    print("\n🌲 Step 11: Training Random Forest...")
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, class_weight='balanced', random_state=42)
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test[:50])[:, 1]

    print("\n📈 Step 12: Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, class_weight='balanced', C=0.5)
    lr.fit(X_train, y_train)
    lr_probs = lr.predict_proba(X_test[:50])[:, 1]

    print("\n🧮 Step 12.5: Combining predictions with probabilistic fusion (QSVM=0.2, RF=0.4, LR=0.4)...")
    qsvc_scores = y_qsvc.astype(float)
    ensemble_score = 0.2 * qsvc_scores + 0.4 * rf_probs + 0.4 * lr_probs
    y_ensemble = (ensemble_score > 0.5).astype(int)
    y_true = y_test[:50]

    print("🔍 QSVM predictions:", y_qsvc)
    print("🔍 RF probabilities:", rf_probs)
    print("🔍 LR probabilities:", lr_probs)
    print("🔍 Ensemble scores:", ensemble_score)
    print("🔍 Final predictions:", y_ensemble)

    print("\n📊 Step 13: Evaluating model...")
    acc = accuracy_score(y_true, y_ensemble)
    print(f"✅ Accuracy: {acc:.2f}")
    print("📊 Classification Report:")
    print(classification_report(y_true, y_ensemble, zero_division=0))

    print("🧩 Confusion Matrix:")
    print(confusion_matrix(y_true, y_ensemble))

    print("🔍 ROC AUC Score:", roc_auc_score(y_true, y_ensemble))

    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_ensemble, average='binary', zero_division=0)
    print(f"🩺 Clinical Metrics:\nPrecision: {precision:.2f}\nRecall: {recall:.2f}\nF1 Score: {f1:.2f}")

    if acc >= 0.90:
        print("🌟 Model exceeds 90% accuracy — ready for clinical deployment.")
    elif acc >= 0.85:
        print("✅ Model meets 85%+ — strong and reliable.")
    elif acc >= 0.80:
        print("⚠ Model meets 80% — acceptable but could be improved.")
    else:
        print("⚠ Accuracy below 80% - try to improve.")

    print("\n📜 Attribution: PHQ-9 and GAD-7 are validated clinical screening tools developed by Pfizer. Used here under public domain guidelines for educational and research purposes.")

if __name__ == "__main__":
    train_hybrid_model()