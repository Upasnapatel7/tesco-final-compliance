# from ml_pipeline import NLPComplianceModel, ImageRiskModel
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
# from PIL import Image
# import numpy as np


# # ================= NLP EVALUATION =================
# def evaluate_nlp():
#     model_obj = NLPComplianceModel()
#     model = model_obj._model

#     texts = [d[0] for d in NLPComplianceModel._TRAINING]
#     labels = [d[1] for d in NLPComplianceModel._TRAINING]

#     preds = model.predict(texts)

#     print("=== NLP MODEL EVALUATION ===")
#     print("Accuracy:", accuracy_score(labels, preds))
#     print("Precision:", precision_score(labels, preds, average='weighted'))
#     print("Recall:", recall_score(labels, preds, average='weighted'))
#     print("F1 Score:", f1_score(labels, preds, average='weighted'))

#     print("\nDetailed Report:\n")
#     print(classification_report(labels, preds))


# # ================= IMAGE EVALUATION =================
# def evaluate_image():
#     model_obj = ImageRiskModel()
#     model = model_obj._model
#     extractor = model_obj._extractor

#     X = []
#     y = []

#     # LOW RISK
#     for _ in range(40):
#         img = Image.new("RGB", (128,128), "white")
#         X.append(extractor.extract(img))
#         y.append("low_risk")

#     # MEDIUM RISK
#     for _ in range(40):
#         img = Image.new("RGB", (128,128), "pink")
#         X.append(extractor.extract(img))
#         y.append("medium_risk")

#     # HIGH RISK
#     for _ in range(40):
#         img = Image.new("RGB", (128,128), "green")
#         X.append(extractor.extract(img))
#         y.append("high_risk")

#     # CRITICAL RISK
#     for _ in range(40):
#         img = Image.new("RGB", (128,128), "red")
#         X.append(extractor.extract(img))
#         y.append("critical_risk")

#     X = np.array(X)

#     preds = model.predict(X)

#     print("\n=== IMAGE MODEL EVALUATION ===")
#     print(classification_report(y, preds))


# # ================= MAIN =================
# if __name__ == "__main__":
#     evaluate_nlp()
#     evaluate_image()

import pandas as pd
import os
import numpy as np

from ml_pipeline import NLPComplianceModel, ImageRiskModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from PIL import Image


# ================= NLP EVALUATION =================
def evaluate_nlp_real():
    print("=== NLP MODEL (REAL EVALUATION) ===")

    csv_path = "nlp_dataset_100.csv"

    if not os.path.exists(csv_path):
        print(f"❌ CSV not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Validate columns
    if "text" not in df.columns or "label" not in df.columns:
        print("❌ CSV must contain 'text' and 'label'")
        print("Found:", df.columns)
        return

    X = df["text"].astype(str)
    y = df["label"].astype(str)

    # ✅ STRATIFIED SPLIT (fix)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.4,
        random_state=42,
        stratify=y
    )

    # Load model
    model_obj = NLPComplianceModel()
    model = model_obj._model

    # Train
    model.fit(X_train, y_train)

    # Predict
    preds = model.predict(X_test)

    # Metrics
    print("Accuracy:", accuracy_score(y_test, preds))
    print("Precision:", precision_score(y_test, preds, average='weighted'))
    print("Recall:", recall_score(y_test, preds, average='weighted'))
    print("F1 Score:", f1_score(y_test, preds, average='weighted'))

    print("\nDetailed Report:\n")
    print(classification_report(y_test, preds))


# ================= IMAGE EVALUATION =================
def evaluate_image_real():
    print("\n=== IMAGE MODEL (REAL DATA EVALUATION) ===")

    model_obj = ImageRiskModel()
    extractor = model_obj._extractor

    base_path = "dataset"

    classes = ["low_risk", "medium_risk", "high_risk", "critical_risk"]

    image_paths = []
    labels = []

    # ✅ LOAD PATHS ONLY (NO AUGMENTATION HERE)
    for label in classes:
        folder = os.path.join(base_path, label)

        if not os.path.exists(folder):
            print(f"⚠️ Missing folder: {folder}")
            continue

        for file in os.listdir(folder):
            image_paths.append(os.path.join(folder, file))
            labels.append(label)

    print("Total real images:", len(image_paths))

    if len(image_paths) < 20:
        print("⚠️ WARNING: Not enough data.")
        return

    # ✅ SPLIT FIRST (fix leakage)
    train_paths, test_paths, y_train, y_test = train_test_split(
        image_paths,
        labels,
        test_size=0.3,
        random_state=42,
        stratify=labels
    )

    # 🔹 Feature extraction AFTER split
    def extract_features(paths):
        features = []
        valid_labels = []

        for path, label in zip(paths, labels):
            try:
                img = Image.open(path).convert("RGB")
                feat = extractor.extract(img)
                features.append(feat)
                valid_labels.append(label)
            except Exception as e:
                print(f"Skipping {path}: {e}")

        return np.array(features), np.array(valid_labels)

    X_train, y_train = extract_features(train_paths)
    X_test, y_test = extract_features(test_paths)

    print("Train samples:", len(X_train))
    print("Test samples:", len(X_test))

    # ✅ TRAIN (clean)
    model_obj.fit_real(X_train, y_train)
    model = model_obj._model

    # ✅ PREDICT (with scaler)
    preds = model.predict(model_obj._scaler.transform(X_test))

    # 🔍 DEBUG
    print("\n--- Predictions Debug ---")
    for i in range(len(y_test)):
        print("Actual:", y_test[i], "| Predicted:", preds[i])

    # 📊 METRICS
    print("\nAccuracy:", round(accuracy_score(y_test, preds), 4))
    print(classification_report(y_test, preds))


# ================= MAIN =================
if __name__ == "__main__":
    evaluate_nlp_real()
    evaluate_image_real()