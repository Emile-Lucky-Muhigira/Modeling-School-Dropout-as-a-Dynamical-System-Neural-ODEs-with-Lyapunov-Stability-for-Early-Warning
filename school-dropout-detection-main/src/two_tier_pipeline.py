"""
Two-Tier Dropout Prediction Pipeline
=====================================
Combines:
- Tier 1: Country-level context (Kaggle Education in Africa)
- Tier 2: Student-level prediction (Young Lives Ethiopia)

This integration captures both individual risk factors AND systemic
educational context for more robust predictions.

Usage:
    python src/two_tier_pipeline.py
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
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
KAGGLE_DIR = DATA_DIR / "raw" / "kaggle"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"

# Create directories
for d in [MODELS_DIR, FIGURES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# =============================================================================
# TIER 1: COUNTRY-LEVEL DATA (KAGGLE)
# =============================================================================

def load_kaggle_data():
    """Load and process Kaggle Education in Africa datasets."""
    
    print("\n" + "=" * 60)
    print("TIER 1: Loading Country-Level Data (Kaggle)")
    print("=" * 60)
    
    # Load all CSV files
    files = {
        'general': 'Education in General.csv',
        'primary': 'Primary_Education.csv',
        'primary_attendance': 'Primary Education Attendance.csv',
        'secondary': 'Secondary Education.csv',
        'tertiary': 'Tertiary Education.csv',
        'population': 'School Age Population.csv',
        'illiterate': 'Illiterate Population.csv',
    }
    
    dfs = {}
    for key, filename in files.items():
        path = KAGGLE_DIR / filename
        if path.exists():
            df = pd.read_csv(path)
            # Replace #N/B with NaN
            df = df.replace('#N/B', np.nan)
            dfs[key] = df
            print(f"  Loaded {filename}: {len(df)} rows")
        else:
            print(f"  WARNING: {filename} not found")
    
    # Merge datasets on Country and Year
    merged = None
    for key, df in dfs.items():
        if 'Country' in df.columns and 'Year' in df.columns:
            # Select relevant columns (avoid duplicates)
            if merged is None:
                merged = df
            else:
                # Get columns that aren't already in merged (except Country, Year)
                new_cols = ['Country', 'Year'] + [c for c in df.columns 
                           if c not in merged.columns and c not in ['Country', 'Year']]
                if len(new_cols) > 2:
                    merged = merged.merge(df[new_cols], on=['Country', 'Year'], how='outer')
    
    print(f"\n  Merged dataset: {len(merged)} rows, {len(merged.columns)} columns")
    
    # Filter to Ethiopia (matching Young Lives data)
    ethiopia = merged[merged['Country'] == 'Ethiopia'].copy()
    print(f"  Ethiopia data: {len(ethiopia)} rows")
    
    return merged, ethiopia


def extract_country_features(country_df):
    """Extract key country-level features for the two-tier model."""
    
    # Convert Year to numeric
    country_df['Year'] = pd.to_numeric(country_df['Year'], errors='coerce')
    
    # Key indicators to extract
    feature_cols = []
    
    # Try to identify useful columns
    for col in country_df.columns:
        col_lower = col.lower()
        if any(term in col_lower for term in [
            'enrol', 'attendance', 'completion', 'survival', 'dropout',
            'teacher', 'pupil', 'expenditure', 'ratio', 'rate', 'out-of-school'
        ]):
            feature_cols.append(col)
    
    # Convert to numeric
    for col in feature_cols:
        country_df[col] = pd.to_numeric(country_df[col], errors='coerce')
    
    print(f"  Extracted {len(feature_cols)} country-level features")
    
    return country_df, feature_cols


def compute_country_risk_score(country_df, feature_cols):
    """Compute a composite country-level risk score."""
    
    # Normalize each feature to 0-1 scale
    risk_components = []
    
    for col in feature_cols:
        if country_df[col].notna().sum() > 0:
            values = country_df[col].copy()
            
            # Determine if higher is worse (risk) or better
            col_lower = col.lower()
            if any(term in col_lower for term in ['dropout', 'out-of-school', 'illiterate']):
                # Higher is worse
                normalized = (values - values.min()) / (values.max() - values.min() + 1e-8)
            elif any(term in col_lower for term in ['enrol', 'attendance', 'completion', 'survival']):
                # Higher is better (invert)
                normalized = 1 - (values - values.min()) / (values.max() - values.min() + 1e-8)
            else:
                continue
            
            risk_components.append(normalized)
    
    if risk_components:
        # Average risk score
        country_df['country_risk_score'] = pd.concat(risk_components, axis=1).mean(axis=1)
    else:
        country_df['country_risk_score'] = 0.5  # Default neutral
    
    return country_df


# =============================================================================
# TIER 2: STUDENT-LEVEL DATA (YOUNG LIVES)
# =============================================================================

def load_student_data():
    """Load Young Lives student-level data."""
    
    print("\n" + "=" * 60)
    print("TIER 2: Loading Student-Level Data (Young Lives)")
    print("=" * 60)
    
    data_path = PROCESSED_DIR / "ethiopia_dropout_panel.csv"
    df = pd.read_csv(data_path)
    
    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Filter to valid targets
    df = df[df['dropout_next_round'].notna()].copy()
    print(f"  After filtering: {len(df)} rows with valid target")
    
    return df


# =============================================================================
# TWO-TIER INTEGRATION
# =============================================================================

def integrate_tiers(student_df, country_df):
    """Integrate country-level context with student-level data."""
    
    print("\n" + "=" * 60)
    print("INTEGRATING TWO TIERS")
    print("=" * 60)
    
    # Map Young Lives rounds to approximate years
    # Round 1: 2002, Round 2: 2006, Round 3: 2009, Round 4: 2013, Round 5: 2016
    round_to_year = {1: 2002, 2: 2006, 3: 2009, 4: 2013, 5: 2016}
    
    student_df['survey_year'] = student_df['round'].map(round_to_year)
    
    # Get Ethiopia's country features by year
    if 'Year' in country_df.columns:
        country_df['Year'] = pd.to_numeric(country_df['Year'], errors='coerce')
        ethiopia_by_year = country_df.set_index('Year')
        
        # Get country risk score for each student based on their survey year
        def get_country_risk(year):
            if pd.isna(year):
                return np.nan
            year = int(year)
            # Find closest available year
            available_years = ethiopia_by_year.index.dropna()
            if len(available_years) == 0:
                return np.nan
            closest_year = min(available_years, key=lambda x: abs(x - year))
            if 'country_risk_score' in ethiopia_by_year.columns:
                return ethiopia_by_year.loc[closest_year, 'country_risk_score']
            return np.nan
        
        student_df['country_risk_score'] = student_df['survey_year'].apply(get_country_risk)
        
        # Fill missing with median
        median_risk = student_df['country_risk_score'].median()
        if pd.isna(median_risk):
            median_risk = 0.5
        student_df['country_risk_score'] = student_df['country_risk_score'].fillna(median_risk)
        
        print(f"  Added country_risk_score to student data")
        print(f"  Mean country risk: {student_df['country_risk_score'].mean():.3f}")
    
    return student_df


# =============================================================================
# FEATURE CONFIGURATION
# =============================================================================

STUDENT_FEATURES = {
    'child': ['chsex', 'agemon', 'chlang'],
    'health': ['stunting', 'underweight', 'chhealth'],
    'education': ['engrade', 'hghgrade', 'preprim', 'hschool', 'literate'],
    'household': ['wi_new', 'hq_new', 'hhsize', 'ownhouse'],
    'parental': ['momedu', 'dadedu', 'caredu', 'momlive', 'dadlive'],
    'shocks': ['shecon1', 'shecon2', 'shenv1', 'shfam1', 'shfam2'],
    'location': ['typesite'],
}

COUNTRY_FEATURES = ['country_risk_score']

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
    'country_risk_score': '🌍 Country Risk Context',
}


# =============================================================================
# MODEL TRAINING
# =============================================================================

def prepare_features(df):
    """Prepare feature matrix."""
    
    # Collect all features
    all_features = []
    for group_features in STUDENT_FEATURES.values():
        all_features.extend(group_features)
    all_features.extend(COUNTRY_FEATURES)
    
    # Keep only available features
    available_features = [f for f in all_features if f in df.columns]
    
    X = df[available_features].copy()
    y = df['dropout_next_round'].astype(int)
    
    # Handle missing values
    X = X.fillna(X.median())
    
    return X, y, available_features


def train_two_tier_model(X_train, y_train, X_val, y_val):
    """Train XGBoost with two-tier features."""
    
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


def compute_saliency(model, X, feature_names):
    """Compute SHAP-based saliency."""
    
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    importance = np.abs(shap_values).mean(axis=0)
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importance,
        'human_name': [FEATURE_NAMES.get(f, f) for f in feature_names]
    }).sort_values('importance', ascending=False)
    
    return feature_importance, shap_values


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_two_tier_importance(feature_importance, save_path):
    """Plot feature importance highlighting country-level features."""
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top_n = feature_importance.head(15).copy()
    top_n = top_n.iloc[::-1]
    
    # Color country-level features differently
    colors = []
    for feat in top_n['feature']:
        if feat in COUNTRY_FEATURES:
            colors.append('#2E86AB')  # Blue for country
        else:
            colors.append('#A23B72')  # Purple for student
    colors = colors[::-1]
    
    bars = ax.barh(
        y=top_n['human_name'],
        width=top_n['importance'],
        color=colors
    )
    
    ax.set_xlabel('Mean |SHAP Value| (Gradient Saliency)', fontsize=12)
    ax.set_title('Two-Tier Dropout Risk Drivers\n(🌍 = Country-Level Context)', 
                 fontsize=14, fontweight='bold')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#A23B72', label='Student-Level (Tier 2)'),
        Patch(facecolor='#2E86AB', label='Country-Level (Tier 1)')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {save_path}")


def plot_tier_comparison(metrics_single, metrics_two_tier, save_path):
    """Compare single-tier vs two-tier performance."""
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    metrics_names = ['AUC-ROC', 'F1-Score', 'Precision', 'Recall']
    single_values = [metrics_single['auc_roc'], metrics_single['f1'], 
                     metrics_single['precision'], metrics_single['recall']]
    two_tier_values = [metrics_two_tier['auc_roc'], metrics_two_tier['f1'],
                       metrics_two_tier['precision'], metrics_two_tier['recall']]
    
    x = np.arange(len(metrics_names))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, single_values, width, label='Student-Only', color='#A23B72')
    bars2 = ax.bar(x + width/2, two_tier_values, width, label='Two-Tier', color='#2E86AB')
    
    ax.set_ylabel('Score')
    ax.set_title('Single-Tier vs Two-Tier Model Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.set_ylim(0, 1)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  Saved: {save_path}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    """Main two-tier training pipeline."""
    
    print("\n" + "=" * 60)
    print("   TWO-TIER DROPOUT PREDICTION PIPELINE")
    print("   Student-Level + Country-Level Context")
    print("=" * 60)
    
    # ===== TIER 1: Country Data =====
    all_countries, ethiopia = load_kaggle_data()
    ethiopia, country_features = extract_country_features(ethiopia)
    ethiopia = compute_country_risk_score(ethiopia, country_features)
    
    # ===== TIER 2: Student Data =====
    student_df = load_student_data()
    
    # ===== INTEGRATE =====
    student_df = integrate_tiers(student_df, ethiopia)
    
    # ===== PREPARE FEATURES =====
    X, y, feature_names = prepare_features(student_df)
    
    print(f"\n  Final feature set: {len(feature_names)} features")
    print(f"  Student features: {len(feature_names) - len(COUNTRY_FEATURES)}")
    print(f"  Country features: {len(COUNTRY_FEATURES)}")
    
    # ===== SPLIT DATA =====
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\n  Data splits:")
    print(f"    Train: {len(X_train):,}")
    print(f"    Val:   {len(X_val):,}")
    print(f"    Test:  {len(X_test):,}")
    
    # ===== TRAIN SINGLE-TIER (baseline) =====
    print("\n" + "-" * 40)
    print("Training Single-Tier Model (baseline)...")
    print("-" * 40)
    
    # Remove country features for baseline
    student_only_features = [f for f in feature_names if f not in COUNTRY_FEATURES]
    X_train_single = X_train[student_only_features]
    X_val_single = X_val[student_only_features]
    X_test_single = X_test[student_only_features]
    
    model_single = train_two_tier_model(X_train_single, y_train, X_val_single, y_val)
    metrics_single, _, _ = evaluate_model(model_single, X_test_single, y_test)
    
    print(f"  AUC-ROC: {metrics_single['auc_roc']:.4f}")
    print(f"  F1:      {metrics_single['f1']:.4f}")
    
    # ===== TRAIN TWO-TIER =====
    print("\n" + "-" * 40)
    print("Training Two-Tier Model...")
    print("-" * 40)
    
    model_two_tier = train_two_tier_model(X_train, y_train, X_val, y_val)
    metrics_two_tier, y_pred, y_prob = evaluate_model(model_two_tier, X_test, y_test)
    
    print(f"  AUC-ROC: {metrics_two_tier['auc_roc']:.4f}")
    print(f"  F1:      {metrics_two_tier['f1']:.4f}")
    
    # ===== COMPARISON =====
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)
    print(f"\n{'Metric':<15} {'Single-Tier':<15} {'Two-Tier':<15} {'Δ':<10}")
    print("-" * 55)
    
    for metric in ['auc_roc', 'f1', 'precision', 'recall']:
        single_val = metrics_single[metric]
        two_tier_val = metrics_two_tier[metric]
        delta = two_tier_val - single_val
        delta_str = f"+{delta:.4f}" if delta > 0 else f"{delta:.4f}"
        print(f"{metric:<15} {single_val:<15.4f} {two_tier_val:<15.4f} {delta_str}")
    
    # ===== SALIENCY =====
    print("\n" + "=" * 60)
    print("GRADIENT SALIENCY (Two-Tier Model)")
    print("=" * 60)
    
    feature_importance, shap_values = compute_saliency(model_two_tier, X_test, feature_names)
    
    print("\n  TOP 10 DROPOUT DRIVERS:")
    print("  " + "-" * 50)
    for _, row in feature_importance.head(10).iterrows():
        tier = "🌍" if row['feature'] in COUNTRY_FEATURES else "👤"
        bar = "█" * int(row['importance'] * 30)
        print(f"  {tier} {row['human_name']:25s} | {bar} {row['importance']:.4f}")
    
    # ===== SAVE OUTPUTS =====
    print("\n" + "=" * 60)
    print("SAVING OUTPUTS")
    print("=" * 60)
    
    # Plots
    plot_two_tier_importance(feature_importance, FIGURES_DIR / "two_tier_feature_importance.png")
    plot_tier_comparison(metrics_single, metrics_two_tier, FIGURES_DIR / "tier_comparison.png")
    
    # Model
    model_path = MODELS_DIR / "xgboost_two_tier.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model_two_tier, f)
    print(f"  Saved: {model_path}")
    
    # Feature importance
    fi_path = REPORTS_DIR / "two_tier_feature_importance.csv"
    feature_importance.to_csv(fi_path, index=False)
    print(f"  Saved: {fi_path}")
    
    # ===== SUMMARY =====
    print("\n" + "=" * 60)
    print("✅ TWO-TIER PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  Two-Tier Model Performance:")
    print(f"    AUC-ROC:   {metrics_two_tier['auc_roc']:.4f}")
    print(f"    F1-Score:  {metrics_two_tier['f1']:.4f}")
    print(f"    Precision: {metrics_two_tier['precision']:.4f}")
    print(f"    Recall:    {metrics_two_tier['recall']:.4f}")
    
    improvement = metrics_two_tier['auc_roc'] - metrics_single['auc_roc']
    print(f"\n  Improvement over single-tier: {improvement:+.4f} AUC-ROC")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
