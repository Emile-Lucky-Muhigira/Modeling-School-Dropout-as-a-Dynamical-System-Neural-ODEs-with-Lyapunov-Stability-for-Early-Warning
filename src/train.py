"""
School Dropout Prediction - Training Script
============================================
Trains XGBoost classifier on Young Lives Ethiopia data
with gradient saliency for explainability.

Usage:
    python src/train.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE
import xgboost as xgb

# Explainability
import shap

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"

for d in [MODELS_DIR, FIGURES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

FEATURE_GROUPS = {
    'child': ['chsex', 'agemon', 'chlang'],
    'health': ['stunting', 'underweight', 'chhealth'],
    'education': ['engrade', 'hghgrade', 'preprim', 'hschool', 'literate'],
    'household': ['wi_new', 'hq_new', 'hhsize', 'ownhouse'],
    'parental': ['momedu', 'dadedu', 'caredu', 'momlive', 'dadlive'],
    'shocks': ['shecon1', 'shecon2', 'shenv1', 'shfam1', 'shfam2'],
    'location': ['typesite'],
}

FEATURE_NAMES = {
    'chsex': 'Gender',
    'agemon': 'Age (months)',
    'chlang': 'Language Score',
    'stunting': 'Stunting Status',
    'underweight': 'Underweight Status',
    'chhealth': 'Health Status',
    'engrade': 'Current Grade',
    'hghgrade': 'Highest Grade Completed',
    'preprim': 'Pre-Primary Attendance',
    'hschool': 'Time at School',
    'literate': 'Literacy',
    'wi_new': 'Wealth Index',
    'hq_new': 'Housing Quality',
    'hhsize': 'Household Size',
    'ownhouse': 'Home Ownership',
    'momedu': "Mother's Education",
    'dadedu': "Father's Education",
    'caredu': "Caregiver's Education",
    'momlive': 'Mother Alive',
    'dadlive': 'Father Alive',
    'shecon1': 'Economic Shock 1',
    'shecon2': 'Economic Shock 2',
    'shenv1': 'Environmental Shock',
    'shfam1': 'Family Shock 1',
    'shfam2': 'Family Shock 2',
    'typesite': 'Urban/Rural',
}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data():
    print("Loading data...")
    data_path = DATA_DIR / "ethiopia_dropout_panel.csv"
    df = pd.read_csv(data_path)
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")

    df = df[df['dropout_next_round'].notna()].copy()
    print(f"  After filtering: {len(df)} rows with valid target")

    all_features = []
    for group_features in FEATURE_GROUPS.values():
        all_features.extend(group_features)

    available_features = [f for f in all_features if f in df.columns]
    print(f"  Using {len(available_features)} features")

    X = df[available_features].copy()
    y = df['dropout_next_round'].astype(int)
    X = X.fillna(X.median())

    print(f"\n  Target distribution:")
    print(f"    No dropout (0): {(y==0).sum()} ({(y==0).mean()*100:.1f}%)")
    print(f"    Dropout    (1): {(y==1).sum()} ({(y==1).mean()*100:.1f}%)")

    return X, y, available_features


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_xgboost(X_train, y_train, X_val, y_val):
    ratio = (y_train == 0).sum() / (y_train == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=1,
        gamma=0.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
        scale_pos_weight=ratio,
        eval_metric='aucpr',
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)
    return model


def find_best_threshold(y_true, y_prob):
    thresholds = np.arange(0.05, 0.60, 0.01)
    f1_scores  = [f1_score(y_true, (y_prob >= t).astype(int)) for t in thresholds]
    return thresholds[np.argmax(f1_scores)]


def evaluate_model(y_test, y_pred, y_prob, label="MODEL"):
    metrics = {
        'auc_roc':   roc_auc_score(y_test, y_prob),
        'f1':        f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall':    recall_score(y_test, y_pred),
    }
    print("\n" + "=" * 60)
    print(f"{label} PERFORMANCE")
    print("=" * 60)
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Stay', 'Dropout']))
    return metrics


# =============================================================================
# EXPLAINABILITY
# =============================================================================

def compute_saliency(model, X, feature_names):
    print("\n" + "=" * 60)
    print("COMPUTING GRADIENT SALIENCY (SHAP)")
    print("=" * 60)

    explainer     = shap.TreeExplainer(model)
    shap_values   = explainer.shap_values(X)
    importance    = np.abs(shap_values).mean(axis=0)

    feature_importance = pd.DataFrame({
        'feature':    feature_names,
        'importance': importance,
        'human_name': [FEATURE_NAMES.get(f, f) for f in feature_names]
    }).sort_values('importance', ascending=False)

    print("\n  TOP 10 DROPOUT DRIVERS (by gradient saliency):")
    print("  " + "-" * 50)
    for _, row in feature_importance.head(10).iterrows():
        bar = "█" * int(row['importance'] * 50)
        print(f"  {row['human_name']:25s} | {bar} {row['importance']:.4f}")

    return feature_importance, shap_values


def generate_student_explanation(model, X, feature_names, student_idx, best_threshold, y_true=None):
    explainer   = shap.TreeExplainer(model)
    student_data = X.iloc[[student_idx]]
    shap_vals   = explainer.shap_values(student_data)[0]
    prob        = model.predict_proba(student_data)[0, 1]

    feature_shap = sorted(zip(feature_names, shap_vals), key=lambda x: abs(x[1]), reverse=True)

    alert = "🔴 HIGH RISK" if prob > best_threshold else ("🟡 WATCH" if prob > best_threshold * 0.6 else "🟢 LOW RISK")
    actual = ""
    if y_true is not None:
        actual_val = y_true.iloc[student_idx]
        actual = f" | Actual: {'DROPPED OUT' if actual_val == 1 else 'STAYED'}"

    print(f"\n  ┌─────────────────────────────────────────────────────────┐")
    print(f"  │ STUDENT #{student_idx:04d}                                         │")
    print(f"  ├─────────────────────────────────────────────────────────┤")
    print(f"  │ Dropout Risk: {prob*100:5.1f}%    Alert: {alert:15s}     │")
    print(f"  │ {actual:55s} │")
    print(f"  ├─────────────────────────────────────────────────────────┤")
    print(f"  │ TOP 3 RISK DRIVERS:                                     │")
    for feat, shap_val in feature_shap[:3]:
        human_name = FEATURE_NAMES.get(feat, feat)
        direction  = "↑ INCREASES risk" if shap_val > 0 else "↓ DECREASES risk"
        print(f"  │   • {human_name:20s} {direction} (∇={shap_val:+.3f})   │")
    print(f"  └─────────────────────────────────────────────────────────┘")

    return prob, feature_shap[:3]


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_feature_importance(feature_importance, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    top_n   = feature_importance.head(15).copy().iloc[::-1]
    colors  = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_n)))[::-1]
    ax.barh(y=top_n['human_name'], width=top_n['importance'], color=colors)
    ax.set_xlabel('Mean |SHAP Value| (Gradient Saliency)', fontsize=12)
    ax.set_title('Top Dropout Risk Drivers\n(Higher = Stronger Influence on Prediction)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Stay', 'Dropout'],
                yticklabels=['Stay', 'Dropout'],
                ax=ax, annot_kws={'size': 16})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_model_comparison(metrics_xgb, metrics_ensemble, save_path):
    fig, ax      = plt.subplots(figsize=(9, 5))
    metric_labels = ['AUC-ROC', 'F1-Score', 'Precision', 'Recall']
    single_vals   = [metrics_xgb['auc_roc'],      metrics_xgb['f1'],
                     metrics_xgb['precision'],     metrics_xgb['recall']]
    ensemble_vals = [metrics_ensemble['auc_roc'],  metrics_ensemble['f1'],
                     metrics_ensemble['precision'], metrics_ensemble['recall']]
    x     = np.arange(len(metric_labels))
    width = 0.35
    bars1 = ax.bar(x - width/2, single_vals,   width, label='XGBoost Only',    color='#A23B72')
    bars2 = ax.bar(x + width/2, ensemble_vals, width, label='Ensemble (Final)', color='#2E86AB')
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('XGBoost vs Ensemble Model Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.set_ylim(0, 1.05)
    for bar in list(bars1) + list(bars2):
        ax.annotate(f'{bar.get_height():.3f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================
def main():
    print("\n" + "=" * 60)
    print("   SCHOOL DROPOUT PREDICTION - TRAINING PIPELINE")
    print("   Deep Learning Indaba 2026")
    print("=" * 60)

    # Load data
    X, y, feature_names = load_data()

    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"\n  Data splits:")
    print(f"    Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

   # =========================================================
    # STEP 1: Aggressive undersampling + SMOTE combined
    # =========================================================
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.pipeline import Pipeline

    print("\n  Applying undersampling + SMOTE...")

    # First undersample majority to 1:2 ratio, then SMOTE minority up to 1:1
    undersample = RandomUnderSampler(
        sampling_strategy=0.5,  # majority reduced to 2x minority
        random_state=42
    )
    oversample = SMOTE(
        sampling_strategy=1.0,  # then minority brought to 1:1
        k_neighbors=5,
        random_state=42
    )
    pipeline = Pipeline([('u', undersample), ('o', oversample)])
    X_train_bal, y_train_bal = pipeline.fit_resample(X_train, y_train)

    print(f"  Before: {len(X_train):,} | Dropout: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
    print(f"  After:  {len(X_train_bal):,} | Dropout: {y_train_bal.sum()} ({y_train_bal.mean()*100:.1f}%)")

    # =========================================================
    # STEP 2: Train XGBoost with heavy minority focus
    # =========================================================
    print("\n  Training XGBoost...")
    ratio = (y_train_bal == 0).sum() / (y_train_bal == 1).sum()
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=3,
        learning_rate=0.01,
        subsample=0.7,
        colsample_bytree=0.6,
        min_child_weight=1,
        gamma=0.0,
        reg_alpha=0.1,
        reg_lambda=0.5,
        scale_pos_weight=ratio,
        eval_metric='aucpr',
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
    )
    xgb_model.fit(X_train_bal, y_train_bal,
                  eval_set=[(X_val, y_val)],
                  verbose=False)

    # =========================================================
    # STEP 3: Train Random Forest with extreme class weight
    # =========================================================
    print("  Training Random Forest...")
    w = (y_train_bal == 0).sum() / (y_train_bal == 1).sum()
    rf_model = RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_leaf=1,
        class_weight={0: 1, 1: w},
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_bal, y_train_bal)

    # =========================================================
    # STEP 4: Train Gradient Boosting
    # =========================================================
    print("  Training Gradient Boosting...")
    gb_model = GradientBoostingClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.01,
        min_samples_leaf=1,
        subsample=0.7,
        random_state=42
    )
    gb_model.fit(X_train_bal, y_train_bal)

    # =========================================================
    # STEP 5: Find best ensemble weights + threshold together
    # =========================================================
    print("\n  Optimizing ensemble weights and threshold...")

    xgb_p = xgb_model.predict_proba(X_val)[:, 1]
    rf_p  = rf_model.predict_proba(X_val)[:, 1]
    gb_p  = gb_model.predict_proba(X_val)[:, 1]

    best_f1      = 0
    best_weights = (0.5, 0.3, 0.2)
    best_thresh  = 0.5

    for w1 in np.arange(0.3, 0.7, 0.1):
        for w2 in np.arange(0.1, 0.5, 0.1):
            w3 = round(1.0 - w1 - w2, 1)
            if w3 <= 0:
                continue
            ens = w1*xgb_p + w2*rf_p + w3*gb_p
            for t in np.arange(0.05, 0.60, 0.01):
                preds = (ens >= t).astype(int)
                score = f1_score(y_val, preds)
                if score > best_f1:
                    best_f1      = score
                    best_weights = (w1, w2, w3)
                    best_thresh  = t

    print(f"  Best weights: XGB={best_weights[0]:.1f} RF={best_weights[1]:.1f} GB={best_weights[2]:.1f}")
    print(f"  Best threshold: {best_thresh:.2f} | Val F1: {best_f1:.4f}")

    # Apply best weights to test set
    xgb_pt = xgb_model.predict_proba(X_test)[:, 1]
    rf_pt  = rf_model.predict_proba(X_test)[:, 1]
    gb_pt  = gb_model.predict_proba(X_test)[:, 1]

    ensemble_prob   = best_weights[0]*xgb_pt + best_weights[1]*rf_pt + best_weights[2]*gb_pt
    y_pred_ensemble = (ensemble_prob >= best_thresh).astype(int)

    # XGBoost alone metrics
    xgb_pred    = (xgb_pt >= best_thresh).astype(int)
    metrics_xgb = evaluate_model(y_test, xgb_pred, xgb_pt, label="XGBOOST")

    # Ensemble metrics
    metrics_ensemble = evaluate_model(
        y_test, y_pred_ensemble, ensemble_prob, label="ENSEMBLE (FINAL)")

    # =========================================================
    # STEP 6: Cross-validation
    # =========================================================
    print("\n" + "=" * 60)
    print("5-FOLD CROSS VALIDATION")
    print("=" * 60)
    cv_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05,
        subsample=0.7, colsample_bytree=0.6, min_child_weight=1,
        scale_pos_weight=ratio, random_state=42, n_jobs=-1,
    )
    skf       = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(cv_model, X, y, cv=skf, scoring='f1')
    print(f"  F1 per fold: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  Mean F1:     {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # =========================================================
    # STEP 7: Saliency + Explanations
    # =========================================================
    feature_importance, shap_values = compute_saliency(xgb_model, X_test, feature_names)

    print("\n" + "=" * 60)
    print("SAMPLE STUDENT EXPLANATIONS")
    print("=" * 60)
    true_positives = np.where((ensemble_prob > best_thresh) & (y_test.values == 1))[0]
    sample_indices = true_positives[:3] if len(true_positives) >= 3 else np.argsort(ensemble_prob)[-3:]
    for idx in sample_indices:
        generate_student_explanation(xgb_model, X_test, feature_names, idx, best_thresh, y_test)

    # =========================================================
    # STEP 8: Save everything
    # =========================================================
    print("\n" + "=" * 60)
    print("SAVING OUTPUTS")
    print("=" * 60)

    plot_feature_importance(feature_importance,    FIGURES_DIR / "feature_importance.png")
    plot_confusion_matrix(y_test, y_pred_ensemble, FIGURES_DIR / "confusion_matrix.png")
    plot_model_comparison(metrics_xgb, metrics_ensemble, FIGURES_DIR / "model_comparison.png")

    for name, mdl in [("xgboost", xgb_model), ("rf", rf_model), ("gb", gb_model)]:
        with open(MODELS_DIR / f"{name}_dropout.pkl", 'wb') as f:
            pickle.dump(mdl, f)
    print("  Saved: all 3 models")

    feature_importance.to_csv(REPORTS_DIR / "feature_importance.csv", index=False)
    pd.DataFrame([metrics_xgb, metrics_ensemble],
                 index=['XGBoost Only', 'Ensemble Final']
                 ).to_csv(REPORTS_DIR / "metrics_comparison.csv")
    print("  Saved: reports")

    # =========================================================
    # FINAL SUMMARY
    # =========================================================
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE — FINAL SUMMARY")
    print("=" * 60)
    print(f"\n  {'Metric':<12} {'XGBoost Only':>15} {'Ensemble Final':>15}")
    print(f"  {'-'*42}")
    for metric in ['auc_roc', 'f1', 'precision', 'recall']:
        label = metric.upper().replace('_', '-')
        print(f"  {label:<12} {metrics_xgb[metric]:>15.4f} {metrics_ensemble[metric]:>15.4f}")
    print(f"\n  5-Fold CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"\n  Outputs saved to: {OUTPUT_DIR}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()