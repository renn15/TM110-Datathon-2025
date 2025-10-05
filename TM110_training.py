# -*- coding: utf-8 -*-

# Import Libraries

import time                          # measure training & prediction time (latency)
import numpy as np                   # numeric ops
import pandas as pd                  # table ops
import matplotlib.pyplot as plt      # all plots

from ucimlrepo import fetch_ucirepo  # official UCI loader for CTG

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    balanced_accuracy_score, f1_score
)
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
import joblib                        # save/load trained models

# Show wide tables & avoid scientific notation so outputs are readable in judging
pd.set_option("display.max_columns", 200)
np.set_printoptions(suppress=True)


# Load Data

# Fetch dataset from UCI repository
cardiotocography = fetch_ucirepo(id=193)

# Data (as pandas DataFrames)
X = cardiotocography.data.features.copy()
y_all = cardiotocography.data.targets.copy()
y_nsp = y_all["NSP"].astype(int)    # 1=Normal, 2=Suspect, 3=Pathologic
y_cls = y_all["CLASS"].astype(int)  # 1..10


# Train/Test Split

# 80/20 split with stratification so class proportions are preserved
Xtr_nsp, Xte_nsp, ytr_nsp, yte_nsp = train_test_split(X, y_nsp, test_size=0.2, stratify=y_nsp)
Xtr_cls, Xte_cls, ytr_cls, yte_cls = train_test_split(X, y_cls, test_size=0.2, stratify=y_cls)


# Evaluation Function

def evaluate_hgb(name, model, Xtr, ytr, Xte, yte, display_labels=None):
    # Latency
    t0 = time.perf_counter(); model.fit(Xtr, ytr); fit_s = time.perf_counter() - t0
    t1 = time.perf_counter(); ypred = model.predict(Xte); pred_ms = (time.perf_counter() - t1) * 1000

    # Imbalance-aware metrics
    ba  = balanced_accuracy_score(yte, ypred)
    f1m = f1_score(yte, ypred, average="macro")

    print(f"\n{name}")
    print(f"Balanced Accuracy: {ba:.4f} | Macro F1: {f1m:.4f} | fit={fit_s:.3f}s | predict={pred_ms:.1f}ms")
    print(classification_report(yte, ypred, target_names=display_labels) if display_labels else
          classification_report(yte, ypred))
    
    cm = confusion_matrix(yte, ypred)
    ConfusionMatrixDisplay(cm, display_labels=display_labels).plot(cmap="Blues")
    plt.title(f"{name} – Confusion Matrix"); plt.show()

    return {"ba":ba, "f1":f1m, "fit_s":fit_s, "pred_ms":pred_ms}, ypred, model


# Gradient Boosting NSP Model

# HGB works well on tabular data.
hgb_nsp = HistGradientBoostingClassifier()

nsp_metrics, nsp_pred, nsp_fit = evaluate_hgb(
    "HistGradientBoosting (NSP)", hgb_nsp,
    Xtr_nsp, ytr_nsp, Xte_nsp, yte_nsp,
    display_labels=["Normal","Suspect","Pathologic"]
)

# Permutation importance
perm = permutation_importance(nsp_fit, Xte_nsp, yte_nsp, n_repeats=5, scoring="balanced_accuracy", n_jobs=-1)
perm_s = pd.Series(perm.importances_mean, index=X.columns).clip(lower=0).sort_values(ascending=False)

# FEWEST features that explain ≥90% of the total importance contribution
total = perm_s.sum() if perm_s.sum() > 0 else 1.0
cum = (perm_s / total).cumsum()
keep = perm_s.loc[cum <= 0.90]
if keep.empty: keep = perm_s.head(1)


# Gradient Boosting CLASS Model

hgb_cls = HistGradientBoostingClassifier()

labels_cls = [str(i) for i in sorted(y_cls.unique())]
cls_metrics, cls_pred, cls_fit = evaluate_hgb(
    "HistGradientBoosting (CLASS 10-class)", hgb_cls,
    Xtr_cls, ytr_cls, Xte_cls, yte_cls,
    display_labels=labels_cls
)

# Permutation importance
permC = permutation_importance(cls_fit, Xte_cls, yte_cls, n_repeats=5, scoring="balanced_accuracy", n_jobs=-1)
permC_s = pd.Series(permC.importances_mean, index=X.columns).clip(lower=0).sort_values(ascending=False)

totalC = permC_s.sum() if permC_s.sum() > 0 else 1.0
cumC = (permC_s / totalC).cumsum()
keepC = permC_s.loc[cumC <= 0.90]
if keepC.empty: keepC = permC_s.head(1)


# Cross-Validation Between Folds

cv = StratifiedKFold(n_splits=5)

# NSP per-fold scores
ba_nsp = cross_val_score(HistGradientBoostingClassifier(), X, y_nsp, scoring="balanced_accuracy", cv=cv, n_jobs=-1)
f1_nsp = cross_val_score(HistGradientBoostingClassifier(), X, y_nsp, scoring="f1_macro",          cv=cv, n_jobs=-1)

# CLASS per-fold scores
ba_cls = cross_val_score(HistGradientBoostingClassifier(), X, y_cls, scoring="balanced_accuracy", cv=cv, n_jobs=-1)
f1_cls = cross_val_score(HistGradientBoostingClassifier(), X, y_cls, scoring="f1_macro",          cv=cv, n_jobs=-1)

# Line plots
plt.figure(figsize=(8,5))
plt.plot(range(1, len(ba_nsp)+1), ba_nsp, marker="o")
plt.ylim(0,1); plt.grid(True, ls="--", alpha=0.6)
plt.title("NSP – HGB Balanced Accuracy across folds")
plt.xlabel("Fold"); plt.ylabel("Balanced Accuracy")
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,5))
plt.plot(range(1, len(f1_nsp)+1), f1_nsp, marker="o")
plt.ylim(0,1); plt.grid(True, ls="--", alpha=0.6)
plt.title("NSP – HGB Macro-F1 across folds")
plt.xlabel("Fold"); plt.ylabel("Macro-F1")
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,5))
plt.plot(range(1, len(ba_cls)+1), ba_cls, marker="o")
plt.ylim(0,1); plt.grid(True, ls="--", alpha=0.6)
plt.title("CLASS – HGB Balanced Accuracy across folds")
plt.xlabel("Fold"); plt.ylabel("Balanced Accuracy")
plt.tight_layout(); plt.show()

plt.figure(figsize=(8,5))
plt.plot(range(1, len(f1_cls)+1), f1_cls, marker="o")
plt.ylim(0,1); plt.grid(True, ls="--", alpha=0.6)
plt.title("CLASS – HGB Macro-F1 across folds")
plt.xlabel("Fold"); plt.ylabel("Macro-F1")
plt.tight_layout(); plt.show()

"""#Bars for the test-set metrics"""

# Bar chart (one model, two metrics)
fig, ax = plt.subplots(figsize=(6,4))
ax.bar(["NSP BA","NSP F1","CLASS BA","CLASS F1"],
       [nsp_metrics["ba"], nsp_metrics["f1"], cls_metrics["ba"], cls_metrics["f1"]])
ax.set_ylim(0,1); ax.set_ylabel("Score"); ax.set_title("HGB – Test Set Metrics")
plt.tight_layout(); plt.show()

"""#Save Models & Predict Functions"""

# Save both trained HGB models so we can load without retraining
joblib.dump(hgb_nsp, "HGB_NSP.pkl")
joblib.dump(hgb_cls, "HGB_CLASS.pkl")

# Save test predictions for auditability
pd.DataFrame({"y_test": yte_nsp.values, "y_pred": nsp_pred}).to_csv("HGB_NSP_test_preds.csv", index=False)
pd.DataFrame({"y_test": yte_cls.values, "y_pred": cls_pred}).to_csv("HGB_CLASS_test_preds.csv", index=False)
print("Saved: HGB_NSP.pkl, HGB_CLASS.pkl, and test prediction CSVs.")

# Predict functions for live demo
def predict_nsp(new_df):
    model = joblib.load("HGB_NSP.pkl")
    preds = model.predict(new_df[X.columns])   # ensure same columns/order
    return pd.Series(preds).map({1:"Normal", 2:"Suspect", 3:"Pathologic"})

def predict_class(new_df):
    model = joblib.load("HGB_CLASS.pkl")
    preds = model.predict(new_df[X.columns])
    return pd.Series(preds)                    # 1..10 labels

# Mini demo
# NSP demo: show predictions vs true labels
print("\nNSP Demo (first 5 test samples):")
nsp_demo_preds = predict_nsp(Xte_nsp.head(5))      # model predictions
nsp_demo_truth = yte_nsp.head(5).map({1:"Normal", 2:"Suspect", 3:"Pathologic"})  # true labels
nsp_demo = pd.DataFrame({"True Label": nsp_demo_truth.values,
                         "Predicted Label": nsp_demo_preds.values})
print(nsp_demo)

# CLASS demo: show predictions vs true labels
print("\nCLASS Demo (first 5 test samples):")
cls_demo_preds = predict_class(Xte_cls.head(5))    # model predictions
cls_demo_truth = yte_cls.head(5)                   # true labels (1..10)
cls_demo = pd.DataFrame({"True Label": cls_demo_truth.values,
                         "Predicted Label": cls_demo_preds.values})
print(cls_demo)


