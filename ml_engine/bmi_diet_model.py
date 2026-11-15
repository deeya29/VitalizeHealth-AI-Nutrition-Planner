"""
ml_engine/bmi_diet_model.py
===========================
VitalizeHealth — BMI-Based Diet Plan · ML Engine
Ported from Combined_Notebook.ipynb (Random Forest + Diet Recommendations)

Steps mirrored from notebook:
  Step 1  : Import Libraries
  Step 2  : Generate / Load Dataset
  Step 3  : BMI Dataset Cleaning
  Step 4  : BMI Auto-Calculation
  Step 5  : EDA (charts saved to static/charts/)
  Step 6  : Diet Recommendations Label Mapping
  Step 7  : Random Forest Classifier Training
  Step 8  : predict_diet_plan() — used by the Flask API
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no GUI needed)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ─────────────────────────────────────────────────────────────────
# STEP 6 — Diet Recommendations Dictionary (from notebook Step 6)
# ─────────────────────────────────────────────────────────────────
DIET_RECOMMENDATIONS = {
    "Underweight": {
        "goal":           "Gain Weight – High Calorie & Protein",
        "focus":          ["Nuts", "Dairy", "Eggs", "Whole Grains", "Oils", "Banana", "Paneer"],
        "avoid":          ["Junk Food", "Excessive Cardio"],
        "daily_calories": "2500–3000",
        "tip":            "Focus on calorie-dense, nutrient-rich foods eaten frequently (5–6 meals/day)."
    },
    "Normal Weight": {
        "goal":           "Maintain Weight – Balanced Diet",
        "focus":          ["Vegetables", "Fruits", "Lean Protein", "Whole Grains"],
        "avoid":          ["Excess Sugar", "Processed Food"],
        "daily_calories": "2000–2500",
        "tip":            "Maintain variety across food groups; aim for 80/20 clean eating."
    },
    "Overweight": {
        "goal":           "Lose Weight – Low Calorie & High Protein",
        "focus":          ["Vegetables", "Fruits", "Lean Protein", "Dal", "Salad"],
        "avoid":          ["Fried Food", "Sugary Drinks", "White Rice"],
        "daily_calories": "1500–2000",
        "tip":            "Create a 400–500 kcal daily deficit; prioritise fibre and protein to stay satiated."
    },
    "Obese Class 1": {
        "goal":           "Lose Weight – Strict Low Calorie",
        "focus":          ["Vegetables", "Lean Protein", "Fruits", "Green Tea"],
        "avoid":          ["Oils", "Dairy (High Fat)", "Refined Grains", "Sweets"],
        "daily_calories": "1200–1800",
        "tip":            "Structured meal timing and a high-protein diet are key for sustainable fat loss."
    },
    "Obese Class 2": {
        "goal":           "Lose Weight – Very Strict",
        "focus":          ["Green Vegetables", "Lean Protein", "Fiber-rich Foods"],
        "avoid":          ["Oils", "Grains", "High Sugar Fruits", "Dairy", "Fried Food"],
        "daily_calories": "1000–1500",
        "tip":            "Consider supervised nutrition coaching alongside this plan for best results."
    },
    "Obese Class 3": {
        "goal":           "Medical Supervision Required",
        "focus":          ["Consult Doctor First", "Vegetables", "Lean Protein"],
        "avoid":          ["All High Calorie Foods"],
        "daily_calories": "800–1200 (Doctor Only)",
        "tip":            "Please consult a physician before starting any diet at this BMI level."
    },
}


# ─────────────────────────────────────────────────────────────────
# STEP 2 — Dataset Generation (mirrors notebook Step 2)
# ─────────────────────────────────────────────────────────────────
def _generate_bmi_dataset(n: int = 741, seed: int = 42) -> pd.DataFrame:
    """Recreates the BMI dataset from the notebook (741 rows, 5 cols)."""
    np.random.seed(seed)
    ages    = np.random.randint(15, 62, n)
    heights = np.round(np.random.uniform(1.46, 2.07, n), 3)
    weights = np.round(np.random.uniform(25.9, 270.0, n), 2)
    bmis    = np.round(weights / (heights ** 2), 6)
    classes = [_classify_bmi(b) for b in bmis]
    return pd.DataFrame({
        "Age": ages, "Height": heights, "Weight": weights,
        "Bmi": bmis, "BmiClass": classes
    })


# ─────────────────────────────────────────────────────────────────
# STEP 3 + 4 — Cleaning & BMI Recalculation (mirrors notebook)
# ─────────────────────────────────────────────────────────────────
def _clean_and_calc(df: pd.DataFrame) -> pd.DataFrame:
    clean = df.copy()
    clean.columns = clean.columns.str.strip().str.lower().str.replace(" ", "_")
    clean.drop_duplicates(inplace=True)
    clean.dropna(inplace=True)
    clean["bmi"] = pd.to_numeric(clean["bmi"], errors="coerce")
    clean.dropna(subset=["bmi"], inplace=True)

    # Recalculate BMI from raw height/weight (Step 4)
    clean["bmi_calculated"] = (clean["weight"] / (clean["height"] ** 2)).round(2)
    clean["bmiclass_calc"]  = clean["bmi_calculated"].apply(_classify_bmi)
    clean.reset_index(drop=True, inplace=True)
    return clean


# ─────────────────────────────────────────────────────────────────
# BMI Classification — mirrors classify_bmi() from notebook
# ─────────────────────────────────────────────────────────────────
def _classify_bmi(b: float) -> str:
    if b < 18.5:  return "Underweight"
    if b < 25.0:  return "Normal Weight"
    if b < 30.0:  return "Overweight"
    if b < 35.0:  return "Obese Class 1"
    if b < 40.0:  return "Obese Class 2"
    return "Obese Class 3"


# ─────────────────────────────────────────────────────────────────
# STEP 5 — EDA Visualizations (saved to static/charts/)
# ─────────────────────────────────────────────────────────────────
def _save_eda_charts(df: pd.DataFrame, charts_dir: str) -> None:
    os.makedirs(charts_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: BMI Category Distribution
    cat_counts = df["bmiclass_calc"].value_counts()
    axes[0].bar(cat_counts.index, cat_counts.values, color="steelblue", edgecolor="black")
    axes[0].set_title("BMI Category Distribution", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("BMI Category")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=30)

    # Plot 2: BMI Value Histogram
    axes[1].hist(df["bmi_calculated"], bins=30, color="coral", edgecolor="black")
    axes[1].set_title("BMI Value Distribution", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("BMI")
    axes[1].set_ylabel("Count")

    # Plot 3: BMI vs Weight scatter
    axes[2].scatter(df["weight"], df["bmi_calculated"],
                    alpha=0.4, color="green", edgecolors="none", s=20)
    axes[2].set_title("BMI vs Weight", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Weight (kg)")
    axes[2].set_ylabel("BMI")

    plt.tight_layout()
    plt.savefig(os.path.join(charts_dir, "eda_overview.png"), dpi=120)
    plt.close()
    print("✅ EDA chart saved → static/charts/eda_overview.png")


# ─────────────────────────────────────────────────────────────────
# STEP 7 — Train Random Forest Classifier (mirrors notebook Step 7)
# ─────────────────────────────────────────────────────────────────
class BMIDietModel:
    """
    Wraps the Random Forest pipeline from notebook Step 7.
    Usage:
        model = BMIDietModel()
        model.train()                       # trains & evaluates
        result = model.predict(25, 165, 72) # age, height_cm, weight_kg
    """

    def __init__(self):
        self.model   = None
        self.scaler  = StandardScaler()
        self.le      = LabelEncoder()
        self.trained = False
        self._accuracy: float = 0.0

    # ── Train ────────────────────────────────────────────────────
    def train(self, charts_dir: str = "static/charts", verbose: bool = True) -> None:
        # Step 2 — load dataset
        raw = _generate_bmi_dataset()
        if verbose:
            print(f"✅ BMI Dataset loaded  — shape: {raw.shape}")

        # Step 3 + 4 — clean & calculate BMI
        df = _clean_and_calc(raw)
        if verbose:
            print(f"✅ BMI Dataset cleaned — shape: {df.shape}")

        # Step 5 — EDA charts
        _save_eda_charts(df, charts_dir)

        # Step 6 — diet label
        df["diet_label"] = df["bmiclass_calc"].map(
            {k: v["goal"] for k, v in DIET_RECOMMENDATIONS.items()}
        )

        # Step 7 — features & target
        features = ["age", "height", "weight", "bmi_calculated"]
        X = df[features]
        y = df["bmiclass_calc"]

        y_enc    = self.le.fit_transform(y)
        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )

        self.model = RandomForestClassifier(
            n_estimators=100, max_depth=10,
            random_state=42, class_weight="balanced"
        )
        self.model.fit(X_train, y_train)
        self.trained = True

        # Evaluate
        y_pred = self.model.predict(X_test)
        self._accuracy = accuracy_score(y_test, y_pred)

        if verbose:
            print(f"\n✅ Model trained  — Accuracy: {self._accuracy * 100:.2f}%")
            print("\nClassification Report:")
            print(classification_report(y_test, y_pred, target_names=self.le.classes_))

        # Save confusion matrix chart
        self._save_confusion_matrix(y_test, y_pred, charts_dir)
        self._save_feature_importance(features, charts_dir)

    # ── Predict (mirrors predict_diet_plan() from notebook Step 8) ─
    def predict(self, age: int, height_cm: float, weight_kg: float) -> dict:
        """
        Input:
            age        – in years
            height_cm  – e.g. 165
            weight_kg  – e.g. 72
        Returns dict with bmi, category, plan, accuracy
        """
        if not self.trained:
            raise RuntimeError("Model not trained. Call train() first.")

        height_m = height_cm / 100
        bmi_val  = round(weight_kg / (height_m ** 2), 2)

        X_input   = self.scaler.transform([[age, height_m, weight_kg, bmi_val]])
        pred_enc  = self.model.predict(X_input)[0]
        category  = self.le.inverse_transform([pred_enc])[0]
        plan      = DIET_RECOMMENDATIONS[category]

        return {
            "bmi":          bmi_val,
            "category":     category,
            "goal":         plan["goal"],
            "focus":        plan["focus"],
            "avoid":        plan["avoid"],
            "daily_calories": plan["daily_calories"],
            "tip":          plan["tip"],
            "model_accuracy": round(self._accuracy * 100, 2),
        }

    # ── Confusion Matrix Chart ───────────────────────────────────
    def _save_confusion_matrix(self, y_test, y_pred, charts_dir):
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=self.le.classes_, yticklabels=self.le.classes_)
        plt.title("Confusion Matrix — BMI Category Prediction", fontsize=13, fontweight="bold")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "confusion_matrix.png"), dpi=120)
        plt.close()
        print("✅ Confusion matrix saved → static/charts/confusion_matrix.png")

    # ── Feature Importance Chart ─────────────────────────────────
    def _save_feature_importance(self, features, charts_dir):
        importances = self.model.feature_importances_
        plt.figure(figsize=(7, 4))
        plt.barh(features, importances, color="teal", edgecolor="black")
        plt.title("Feature Importance — Random Forest", fontsize=13, fontweight="bold")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        plt.savefig(os.path.join(charts_dir, "feature_importance.png"), dpi=120)
        plt.close()
        print("✅ Feature importance saved → static/charts/feature_importance.png")

    @property
    def accuracy(self) -> float:
        return round(self._accuracy * 100, 2)


# ─────────────────────────────────────────────────────────────────
# Singleton — one trained model instance shared by Flask
# ─────────────────────────────────────────────────────────────────
_model_instance: BMIDietModel | None = None

def get_model() -> BMIDietModel:
    global _model_instance
    if _model_instance is None:
        print("🔄 Training ML model on startup...")
        _model_instance = BMIDietModel()
        _model_instance.train(charts_dir=os.path.join("static", "charts"), verbose=True)
        print(f"🚀 Model ready — accuracy: {_model_instance.accuracy}%\n")
    return _model_instance
