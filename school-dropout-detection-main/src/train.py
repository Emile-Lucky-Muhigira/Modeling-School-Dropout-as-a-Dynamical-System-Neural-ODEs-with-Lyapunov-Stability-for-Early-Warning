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

# ML
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
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

# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Feature groups
FEATURE_GROUPS = {
    'child': ['chsex', 'agemon', 'chlang'],
    'health': ['stunting', 'underweight', 'chhealth'],
    'education': ['engrade', 'hghgrade', 'preprim', 'hschool', 'literate'],
    'household': ['wi_new', 'hq_new', 'hhsize', 'ownhouse'],
    'parental': ['momedu', 'dadedu', 'caredu', 'momlive', 'dadlive'],
    'shocks': ['shecon1', 'shecon2', 'shenv1', 'shfam1', 'shfam2'],
    'location': ['typesite'],
}

# Human-readable feature names for saliency reports
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
    """Load and prepare the dropout prediction dataset."""
    print("Loading data...")
    
    data_path = DATA_DIR / "ethiopia_dropout_panel.csv"
    df = pd.read_csv(data_path)
    
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Filter to rows with valid target
    df = df[df['dropout_next_round'].notna()].copy()
    print(f"  After filtering: {len(df)} rows with valid target")
    
    # Get features
    all_features = []
    for group_features in FEATURE_GROUPS.values():
        all_features.extend(group_features)
    
    # Keep only available features
    available_features = [f for f in all_features if f in df.columns]
    print(f"  Using {len(available_features)} features")
    
    X = df[available_features].copy()
    y = df['dropout_next_round'].astype(int)
    
    # Handle missing values
    X = X.fillna(X.median())
    
    print(f"\n  Target distribution:")
    print(f"    No dropout (0): {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
    print(f"    Dropout (1): {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
    
    return X, y, available_features


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost classifier with early stopping."""
    
    # Handle class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc',
        early_stopping_rounds=20,
        random_state=42,
        n_jobs=-1,
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    return model


def evaluate_model(model, X_test, y_test):
    """Evaluate model and generate metrics."""
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'auc_roc': roc_auc_score(y_test, y_prob),
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
    }
    
    print("\n" + "=" * 60)
    print("MODEL PERFORMANCE")
    print("=" * 60)
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Stay', 'Dropout']))
    
    return metrics, y_pred, y_prob


# =============================================================================
# EXPLAINABILITY - GRADIENT SALIENCY
# =============================================================================

def compute_saliency(model, X, feature_names):
    """Compute SHAP-based saliency for feature importance."""
    
    print("\n" + "=" * 60)
    print("COMPUTING GRADIENT SALIENCY (SHAP)")
    print("=" * 60)
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Mean absolute SHAP value per feature
    importance = np.abs(shap_values).mean(axis=0)
    
    # Create ranking
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance,
        'human_name': [FEATURE_NAMES.get(f, f) for f in feature_names]
    }).sort_values('importance', ascending=False)
    
    print("\n  TOP 10 DROPOUT DRIVERS (by gradient saliency):")
    print("  " + "-" * 50)
    for _, row in feature_importance.head(10).iterrows():
        bar = "█" * int(row['importance'] * 50)
        print(f"  {row['human_name']:25s} | {bar} {row['importance']:.4f}")
    
    return feature_importance, shap_values


def generate_student_explanation(model, X, feature_names, student_idx, y_true=None):
    """Generate individual student explanation."""
    
    explainer = shap.TreeExplainer(model)
    student_data = X.iloc[[student_idx]]
    shap_values = explainer.shap_values(student_data)[0]
    
    # Get prediction
    prob = model.predict_proba(student_data)[0, 1]
    
    # Get top drivers
    feature_shap = list(zip(feature_names, shap_values))
    feature_shap.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Determine alert level
    if prob > 0.5:
        alert = "🔴 HIGH RISK"
    elif prob > 0.3:
        alert = "🟡 WATCH"
    else:
        alert = "🟢 LOW RISK"
    
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
        if shap_val > 0:
            direction = "↑ INCREASES risk"
        else:
            direction = "↓ DECREASES risk"
        print(f"  │   • {human_name:20s} {direction} (∇={shap_val:+.3f})   │")
    
    print(f"  └─────────────────────────────────────────────────────────┘")
    
    return prob, feature_shap[:3]


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_feature_importance(feature_importance, save_path):
    """Plot feature importance chart."""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top_n = feature_importance.head(15).copy()
    top_n = top_n.iloc[::-1]  # Reverse for horizontal bar
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_n)))[::-1]
    
    ax.barh(
        y=top_n['human_name'],
        width=top_n['importance'],
        color=colors
    )
    
    ax.set_xlabel('Mean |SHAP Value| (Gradient Saliency)', fontsize=12)
    ax.set_title('Top Dropout Risk Drivers\n(Higher = Stronger Influence on Prediction)', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {save_path}")


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix."""
    
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Stay', 'Dropout'],
        yticklabels=['Stay', 'Dropout'],
        ax=ax,
        annot_kws={'size': 16}
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {save_path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main training pipeline."""
    
    print("\n" + "=" * 60)
    print("   SCHOOL DROPOUT PREDICTION - TRAINING PIPELINE")
    print("   Deep Learning Indaba 2026")
    print("=" * 60)
    
    # Load data
    X, y, feature_names = load_data()
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\n  Data splits:")
    print(f"    Train: {len(X_train):,} samples ({len(X_train)/len(X)*100:.1f}%)")
    print(f"    Val:   {len(X_val):,} samples ({len(X_val)/len(X)*100:.1f}%)")
    print(f"    Test:  {len(X_test):,} samples ({len(X_test)/len(X)*100:.1f}%)")
    
    # Train model
    print("\n  Training XGBoost classifier...")
    model = train_xgboost(X_train, y_train, X_val, y_val)
    print("  Done!")
    
    # Evaluate
    metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test)
    
    # Compute saliency
    feature_importance, shap_values = compute_saliency(model, X_test, feature_names)
    
    # Generate student explanations
    print("\n" + "=" * 60)
    print("SAMPLE STUDENT EXPLANATIONS")
    print("=" * 60)
    
    # Find high-risk students who actually dropped out (true positives)
    high_risk_mask = y_prob > 0.5
    actual_dropout_mask = y_test == 1
    true_positives = np.where(high_risk_mask & actual_dropout_mask.values)[0]
    
    if len(true_positives) >= 3:
        sample_indices = true_positives[:3]
    else:
        # Fall back to highest risk students
        sample_indices = np.argsort(y_prob)[-3:]
    
    for idx in sample_indices:
        generate_student_explanation(model, X_test, feature_names, idx, y_test)
    
    # Save plots
    print("\n" + "=" * 60)
    print("SAVING OUTPUTS")
    print("=" * 60)
    
    plot_feature_importance(feature_importance, FIGURES_DIR / "feature_importance.png")
    plot_confusion_matrix(y_test, y_pred, FIGURES_DIR / "confusion_matrix.png")
    
    # Save model
    model_path = MODELS_DIR / "xgboost_dropout.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  Saved: {model_path}")
    
    # Save feature importance
    fi_path = REPORTS_DIR / "feature_importance.csv"
    feature_importance.to_csv(fi_path, index=False)
    print(f"  Saved: {fi_path}")
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE")
    print("=" * 60)
    print(f"\n  Final Metrics:")
    print(f"    AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"    F1-Score:  {metrics['f1']:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall:    {metrics['recall']:.4f}")
    print(f"\n  Outputs saved to: {OUTPUT_DIR}")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
