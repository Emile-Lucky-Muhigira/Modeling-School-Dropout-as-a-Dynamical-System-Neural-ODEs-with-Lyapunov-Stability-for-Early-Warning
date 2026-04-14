"""
Multi-Country Two-Tier Dropout Prediction
==========================================
Trains on DHS data from 5 African countries:
Ethiopia, Kenya, Rwanda, Tanzania, Uganda

Two-tier architecture:
- Tier 1: Country-level context (dropout rates, education indicators)
- Tier 2: Individual-level prediction (demographics, education, wealth)

Usage:
    python src/multicountry_model.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle

# ML
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
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

# Feature configuration
INDIVIDUAL_FEATURES = [
    'age', 'female', 'urban', 'education_years', 
    'overage', 'overage_severe', 'wealth_index'
]

COUNTRY_FEATURES = [
    'country_dropout_rate', 'country_enrollment_rate',
    'country_never_enrolled_rate', 'country_mean_education'
]

FEATURE_NAMES = {
    'age': 'Age',
    'female': 'Female',
    'urban': 'Urban Residence',
    'education_years': 'Years of Education',
    'overage': 'Years Overage',
    'overage_severe': 'Severely Overage (2+ yrs)',
    'wealth_index': 'Wealth Index',
    'country_dropout_rate': '🌍 Country Dropout Rate',
    'country_enrollment_rate': '🌍 Country Enrollment Rate',
    'country_never_enrolled_rate': '🌍 Country Never-Enrolled Rate',
    'country_mean_education': '🌍 Country Mean Education',
}


# =============================================================================
# DATA LOADING & TIER 1 FEATURES
# =============================================================================

def load_dhs_data():
    """Load multi-country DHS data."""
    
    print("\n" + "=" * 60)
    print("LOADING DHS MULTI-COUNTRY DATA")
    print("=" * 60)
    
    data_path = DATA_DIR / "dhs_combined_education.csv"
    df = pd.read_csv(data_path)
    
    print(f"  Loaded: {len(df):,} observations")
    print(f"  Countries: {df['country'].nunique()}")
    print(f"  Dropout cases: {df['dropout'].sum():,} ({df['dropout'].mean()*100:.1f}%)")
    
    return df


def add_country_features(df):
    """Add Tier 1 country-level aggregated features."""
    
    print("\n" + "=" * 60)
    print("TIER 1: Adding Country-Level Context")
    print("=" * 60)
    
    # Compute country-level statistics
    country_stats = df.groupby('country').agg({
        'dropout': 'mean',
        'enrolled': 'mean',
        'never_enrolled': 'mean',
        'education_years': 'mean',
    }).rename(columns={
        'dropout': 'country_dropout_rate',
        'enrolled': 'country_enrollment_rate',
        'never_enrolled': 'country_never_enrolled_rate',
        'education_years': 'country_mean_education',
    })
    
    print("\n  Country-Level Statistics:")
    print("  " + "-" * 55)
    print(f"  {'Country':<12} {'Dropout':>10} {'Enrolled':>10} {'Mean Edu':>10}")
    print("  " + "-" * 55)
    
    for country in country_stats.index:
        stats = country_stats.loc[country]
        print(f"  {country.capitalize():<12} {stats['country_dropout_rate']*100:>9.1f}% "
              f"{stats['country_enrollment_rate']*100:>9.1f}% "
              f"{stats['country_mean_education']:>9.1f}")
    
    # Merge back to individual data
    df = df.merge(country_stats, on='country', how='left')
    
    print(f"\n  Added {len(COUNTRY_FEATURES)} country-level features")
    
    return df


# =============================================================================
# MODEL TRAINING
# =============================================================================

def prepare_features(df, include_country=True):
    """Prepare feature matrix."""
    
    if include_country:
        feature_cols = INDIVIDUAL_FEATURES + COUNTRY_FEATURES
    else:
        feature_cols = INDIVIDUAL_FEATURES
    
    # Keep available columns
    available = [c for c in feature_cols if c in df.columns]
    
    X = df[available].copy()
    y = df['dropout'].astype(int)
    
    # Handle missing values
    X = X.fillna(X.median())
    
    return X, y, available


def train_model(X_train, y_train, X_val, y_val):
    """Train XGBoost classifier."""
    
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    
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
    """Evaluate model performance."""
    
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'auc_roc': roc_auc_score(y_test, y_prob),
        'f1': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
    }
    
    return metrics, y_pred, y_prob


def cross_country_validation(df, include_country=True):
    """Leave-one-country-out cross-validation."""
    
    print("\n" + "=" * 60)
    print("LEAVE-ONE-COUNTRY-OUT CROSS-VALIDATION")
    print("=" * 60)
    
    results = []
    countries = df['country'].unique()
    
    for test_country in countries:
        # Split by country
        train_df = df[df['country'] != test_country]
        test_df = df[df['country'] == test_country]
        
        # Prepare features
        X_train, y_train, features = prepare_features(train_df, include_country)
        X_test, y_test, _ = prepare_features(test_df, include_country)
        
        # Split train into train/val
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
        )
        
        # Train
        model = train_model(X_tr, y_tr, X_val, y_val)
        
        # Evaluate
        metrics, _, _ = evaluate_model(model, X_test, y_test)
        metrics['test_country'] = test_country
        metrics['n_test'] = len(test_df)
        results.append(metrics)
        
        print(f"  {test_country.capitalize():12s}: AUC={metrics['auc_roc']:.3f}, "
              f"F1={metrics['f1']:.3f}, Recall={metrics['recall']:.3f}")
    
    # Average performance
    avg_auc = np.mean([r['auc_roc'] for r in results])
    avg_f1 = np.mean([r['f1'] for r in results])
    print(f"\n  {'AVERAGE':12s}: AUC={avg_auc:.3f}, F1={avg_f1:.3f}")
    
    return results


# =============================================================================
# EXPLAINABILITY
# =============================================================================

def compute_saliency(model, X, feature_names):
    """Compute SHAP-based saliency."""
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    importance = np.abs(shap_values).mean(axis=0)
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance,
        'human_name': [FEATURE_NAMES.get(f, f) for f in feature_names],
        'tier': ['Country' if f in COUNTRY_FEATURES else 'Individual' for f in feature_names]
    }).sort_values('importance', ascending=False)
    
    return feature_importance, shap_values


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_country_comparison(results_single, results_twotier, save_path):
    """Compare single-tier vs two-tier by country."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    countries = [r['test_country'].capitalize() for r in results_single]
    single_auc = [r['auc_roc'] for r in results_single]
    twotier_auc = [r['auc_roc'] for r in results_twotier]
    
    x = np.arange(len(countries))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, single_auc, width, label='Individual Only', color='#A23B72')
    bars2 = ax.bar(x + width/2, twotier_auc, width, label='Two-Tier', color='#2E86AB')
    
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_xlabel('Test Country (Leave-One-Out)', fontsize=12)
    ax.set_title('Cross-Country Generalization: Single-Tier vs Two-Tier', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(countries)
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    
    # Add improvement annotations
    for i, (s, t) in enumerate(zip(single_auc, twotier_auc)):
        diff = t - s
        color = 'green' if diff > 0 else 'red'
        ax.annotate(f'{diff:+.3f}', xy=(x[i] + width/2, t + 0.02),
                    ha='center', fontsize=9, color=color, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {save_path}")


def plot_feature_importance_twotier(feature_importance, save_path):
    """Plot feature importance with tier highlighting."""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top_n = feature_importance.head(12).copy()
    top_n = top_n.iloc[::-1]
    
    colors = ['#2E86AB' if t == 'Country' else '#A23B72' for t in top_n['tier']][::-1]
    
    ax.barh(
        y=top_n['human_name'],
        width=top_n['importance'],
        color=colors
    )
    
    ax.set_xlabel('Mean |SHAP Value| (Gradient Saliency)', fontsize=12)
    ax.set_title('Two-Tier Dropout Risk Drivers\nMulti-Country Model (5 African Countries)', 
                 fontsize=14, fontweight='bold')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#A23B72', label='Individual (Tier 2)'),
        Patch(facecolor='#2E86AB', label='Country Context (Tier 1)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {save_path}")


def plot_dropout_by_country(df, save_path):
    """Visualize dropout rates by country."""
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    country_stats = df.groupby('country').agg({
        'dropout': 'mean',
        'enrolled': 'mean',
        'never_enrolled': 'mean'
    }).sort_values('dropout', ascending=True)
    
    countries = [c.capitalize() for c in country_stats.index]
    
    x = np.arange(len(countries))
    width = 0.25
    
    ax.barh(x - width, country_stats['enrolled']*100, width, label='Enrolled', color='#2E86AB')
    ax.barh(x, country_stats['dropout']*100, width, label='Dropout', color='#E74C3C')
    ax.barh(x + width, country_stats['never_enrolled']*100, width, label='Never Enrolled', color='#95A5A6')
    
    ax.set_xlabel('Percentage (%)', fontsize=12)
    ax.set_ylabel('Country', fontsize=12)
    ax.set_title('Education Status by Country\n(DHS School-Age Population, Ages 6-18)', 
                 fontsize=14, fontweight='bold')
    ax.set_yticks(x)
    ax.set_yticklabels(countries)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {save_path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main multi-country modeling pipeline."""
    
    print("\n" + "=" * 60)
    print("   MULTI-COUNTRY TWO-TIER DROPOUT MODEL")
    print("   5 African Countries | DHS Data | 146K Students")
    print("=" * 60)
    
    # Load data
    df = load_dhs_data()
    
    # Add country-level features (Tier 1)
    df = add_country_features(df)
    
    # =========== CROSS-VALIDATION COMPARISON ===========
    print("\n" + "=" * 60)
    print("COMPARING SINGLE-TIER vs TWO-TIER")
    print("=" * 60)
    
    print("\n--- Single-Tier (Individual Features Only) ---")
    results_single = cross_country_validation(df, include_country=False)
    
    print("\n--- Two-Tier (Individual + Country Context) ---")
    results_twotier = cross_country_validation(df, include_country=True)
    
    # =========== FULL MODEL TRAINING ===========
    print("\n" + "=" * 60)
    print("TRAINING FULL TWO-TIER MODEL")
    print("=" * 60)
    
    X, y, feature_names = prepare_features(df, include_country=True)
    
    # Split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\n  Train: {len(X_train):,}")
    print(f"  Val:   {len(X_val):,}")
    print(f"  Test:  {len(X_test):,}")
    
    # Train
    model = train_model(X_train, y_train, X_val, y_val)
    
    # Evaluate
    metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test)
    
    print("\n" + "=" * 60)
    print("FINAL MODEL PERFORMANCE")
    print("=" * 60)
    print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    
    # Saliency
    print("\n" + "=" * 60)
    print("GRADIENT SALIENCY")
    print("=" * 60)
    
    feature_importance, shap_values = compute_saliency(model, X_test, feature_names)
    
    print("\n  TOP DROPOUT DRIVERS:")
    print("  " + "-" * 55)
    for _, row in feature_importance.head(10).iterrows():
        tier_icon = "🌍" if row['tier'] == 'Country' else "👤"
        print(f"  {tier_icon} {row['human_name']:30s} | {row['importance']:.4f}")
    
    # =========== SAVE OUTPUTS ===========
    print("\n" + "=" * 60)
    print("SAVING OUTPUTS")
    print("=" * 60)
    
    # Plots
    plot_dropout_by_country(df, FIGURES_DIR / "dhs_dropout_by_country.png")
    plot_feature_importance_twotier(feature_importance, FIGURES_DIR / "multicountry_feature_importance.png")
    plot_country_comparison(results_single, results_twotier, FIGURES_DIR / "country_generalization.png")
    
    # Model
    model_path = MODELS_DIR / "xgboost_multicountry.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"  Saved: {model_path}")
    
    # Feature importance
    fi_path = REPORTS_DIR / "multicountry_feature_importance.csv"
    feature_importance.to_csv(fi_path, index=False)
    print(f"  Saved: {fi_path}")
    
    # Cross-validation results
    cv_results = pd.DataFrame(results_twotier)
    cv_path = REPORTS_DIR / "cross_country_validation.csv"
    cv_results.to_csv(cv_path, index=False)
    print(f"  Saved: {cv_path}")
    
    # =========== SUMMARY ===========
    print("\n" + "=" * 60)
    print("✅ MULTI-COUNTRY MODEL COMPLETE")
    print("=" * 60)
    
    avg_single = np.mean([r['auc_roc'] for r in results_single])
    avg_twotier = np.mean([r['auc_roc'] for r in results_twotier])
    improvement = avg_twotier - avg_single
    
    print(f"\n  Cross-Country Generalization:")
    print(f"    Single-Tier AUC: {avg_single:.4f}")
    print(f"    Two-Tier AUC:    {avg_twotier:.4f}")
    print(f"    Improvement:     {improvement:+.4f}")
    
    print(f"\n  Full Model Test Performance:")
    print(f"    AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"    F1-Score:  {metrics['f1']:.4f}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
