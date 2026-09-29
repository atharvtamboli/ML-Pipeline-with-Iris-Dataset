# ─────────────────────────────────────────────────────────────
#  Task 17 – Pipeline Creation  (Iris Dataset)
#  Combines: SimpleImputer → StandardScaler → RandomForestClassifier
# ─────────────────────────────────────────────────────────────

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# ── 1. Load the Iris Dataset ─────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 – Load Iris Dataset")
print("=" * 60)

iris = load_iris()
X, y = iris.data, iris.target

print(f"  Dataset     : Iris (Fisher, 1936)")
print(f"  Samples     : {X.shape[0]}")
print(f"  Features    : {X.shape[1]}  →  {list(iris.feature_names)}")
print(f"  Classes     : {list(iris.target_names)}")
print(f"  Class counts: { {name: (y == i).sum() for i, name in enumerate(iris.target_names)} }")

# Introduce ~8% missing values so the Imputer has work to do
rng = np.random.default_rng(0)
mask = rng.random(X.shape) < 0.08
X[mask] = np.nan
print(f"  NaN cells   : {np.isnan(X).sum()} ({np.isnan(X).mean()*100:.1f}% of all values)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train / test: {len(X_train)} / {len(X_test)}")

# ── 2. Build the Pipeline ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 – Build the Pipeline")
print("=" * 60)

pipeline = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler",  StandardScaler()),
    ("model",   RandomForestClassifier(n_estimators=100, random_state=42)),
])

print("  Pipeline steps:")
for name, step in pipeline.steps:
    print(f"    [{name}]  →  {step.__class__.__name__}")

# ── 3. Train ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 – Train  (pipeline.fit)")
print("=" * 60)

pipeline.fit(X_train, y_train)
print("  ✓ Pipeline trained on Iris training data.")
print("    • Imputer  : filled NaNs using column means from X_train")
print("    • Scaler   : normalised features (mean=0, std=1)")
print("    • RandomForest : fitted 100 trees on scaled data")

# ── 4. Predict ───────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 – Predict  (pipeline.predict)")
print("=" * 60)

y_pred = pipeline.predict(X_test)

print(f"  {'#':<5} {'Actual':<15} {'Predicted':<15} {'Match'}")
print(f"  {'-'*45}")
for i, (actual, pred) in enumerate(zip(y_test, y_pred)):
    match = "✓" if actual == pred else "✗"
    print(f"  {i:<5} {iris.target_names[actual]:<15} {iris.target_names[pred]:<15} {match}")

# ── 5. Evaluate ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 – Evaluate Results")
print("=" * 60)

acc = accuracy_score(y_test, y_pred)
print(f"  Accuracy : {acc:.4f}  ({acc*100:.2f}%)\n")
print("  Classification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# ── 6. Predict probabilities for first 5 test samples ───────────────────────
print("=" * 60)
print("STEP 6 – Predict Probabilities (first 5 samples)")
print("=" * 60)

proba = pipeline.predict_proba(X_test[:5])
header = f"  {'#':<5} " + "".join(f"{n:<14}" for n in iris.target_names) + "Predicted"
print(header)
print(f"  {'-'*55}")
for i, (p, pred) in enumerate(zip(proba, y_pred[:5])):
    row = f"  {i:<5} " + "".join(f"{v:<14.4f}" for v in p) + iris.target_names[pred]
    print(row)

print("\n  ✓ All done.")

# ── 7. Confusion Matrix ──────────────────────────────────────────────────────
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
classes = iris.target_names

plt.figure()
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.colorbar()

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes)
plt.yticks(tick_marks, classes)

# Add numbers inside boxes
for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.ylabel("Actual")
plt.xlabel("Predicted")

plt.tight_layout()
plt.show()

# ── 8. Individual Metrics ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8 – Individual Metrics Explained")
print("=" * 60)

from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_test, y_pred, average=None)
recall    = recall_score(y_test, y_pred, average=None)
f1        = f1_score(y_test, y_pred, average=None)

print(f"\n  {'Class':<14} {'Precision':<14} {'Recall':<14} {'F1-Score'}")
print(f"  {'-'*52}")
for i, name in enumerate(classes):
    print(f"  {name:<14} {precision[i]:<14.2f} {recall[i]:<14.2f} {f1[i]:.2f}")

print(f"\n  Overall Accuracy : {accuracy_score(y_test, y_pred):.2f}")
print(f"  Macro Avg F1     : {f1_score(y_test, y_pred, average='macro'):.2f}")
print("""
  What each metric means:
  • Precision  = Of all flowers predicted as X, how many were truly X?
  • Recall     = Of all actual X flowers, how many did we correctly catch?
  • F1-Score   = Balance between Precision and Recall (higher = better)
  • Accuracy   = Overall correct predictions out of total
""")