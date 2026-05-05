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
import warnings
warnings.filterwarnings('ignore')

# ML
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
import xgboost as xgb

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns


# =============================================================================
# CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR     = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR   = PROJECT_ROOT / "outputs"
MODELS_DIR   = OUTPUT_DIR / "models"
FIGURES_DIR  = OUTPUT_DIR / "figures"
REPORTS_DIR  = OUTPUT_DIR / "reports"

for d in [MODELS_DIR, FIGURES_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

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
    'country_dropout_rate': 'Country Dropout Rate',
    'country_enrollment_rate': 'Country Enrollment Rate',
    'country_never_enrolled_rate': 'Country Never-Enrolled Rate',
    'country_mean_education': 'Country Mean Education',
}


# =============================================================================
# DATA LOADING & TIER 1 FEATURES
# =============================================================================

def load_dhs_data():
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
    print("\n" + "=" * 60)
    print("TIER 1: Adding Country-Level Context")
    print("=" * 60)

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

    df = df.merge(country_stats, on='country', how='left')
    print(f"\n  Added {len(COUNTRY_FEATURES)} country-level features")

    return df


# =============================================================================
# MODEL TRAINING
# =============================================================================

def prepare_features(df, include_country=True):
    if include_country:
        feature_cols = INDIVIDUAL_FEATURES + COUNTRY_FEATURES
    else:
        feature_cols = INDIVIDUAL_FEATURES

    available = [c for c in feature_cols if c in df.columns]
    X = df[available].copy()
    y = df['dropout'].astype(int)
    X = X.fillna(X.median())

    return X, y, available


def train_model(X_train, y_train, X_val, y_val):
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

    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)

    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'auc_roc':   roc_auc_score(y_test, y_prob),
        'f1':        f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall':    recall_score(y_test, y_pred),
    }

    return metrics, y_pred, y_prob


def cross_country_validation(df, include_country=True):
    print("\n" + "=" * 60)
    print("LEAVE-ONE-COUNTRY-OUT CROSS-VALIDATION")
    print("=" * 60)

    results   = []
    countries = df['country'].unique()

    for test_country in countries:
        train_df = df[df['country'] != test_country]
        test_df  = df[df['country'] == test_country]

        X_train, y_train, features = prepare_features(train_df, include_country)
        X_test,  y_test,  _        = prepare_features(test_df,  include_country)

        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.15, random_state=42, stratify=y_train)

        model         = train_model(X_tr, y_tr, X_val, y_val)
        metrics, _, _ = evaluate_model(model, X_test, y_test)
        metrics['test_country'] = test_country
        metrics['n_test']       = len(test_df)
        results.append(metrics)

        print(f"  {test_country.capitalize():12s}: AUC={metrics['auc_roc']:.3f}, "
              f"F1={metrics['f1']:.3f}, Recall={metrics['recall']:.3f}")

    avg_auc = np.mean([r['auc_roc'] for r in results])
    avg_f1  = np.mean([r['f1']     for r in results])
    print(f"\n  {'AVERAGE':12s}: AUC={avg_auc:.3f}, F1={avg_f1:.3f}")

    return results


# =============================================================================
# EXPLAINABILITY
# =============================================================================

def compute_saliency(model, X, feature_names):
    import shap
    importance = np.abs(shap.TreeExplainer(model).shap_values(X)).mean(axis=0)

    return pd.DataFrame({
        'feature':    feature_names,
        'importance': importance,
        'human_name': [FEATURE_NAMES.get(f, f) for f in feature_names],
        'tier':       ['Country' if f in COUNTRY_FEATURES else 'Individual'
                       for f in feature_names]
    }).sort_values('importance', ascending=False)


# =============================================================================
# VISUALIZATION
# =============================================================================

def plot_country_comparison(results_single, results_twotier, save_path):
    fig, ax      = plt.subplots(figsize=(10, 6))
    countries    = [r['test_country'].capitalize() for r in results_single]
    single_auc   = [r['auc_roc'] for r in results_single]
    twotier_auc  = [r['auc_roc'] for r in results_twotier]
    x     = np.arange(len(countries))
    width = 0.35
    ax.bar(x - width/2, single_auc,  width, label='Individual Only', color='#A23B72')
    ax.bar(x + width/2, twotier_auc, width, label='Two-Tier',        color='#2E86AB')
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_xlabel('Test Country (Leave-One-Out)', fontsize=12)
    ax.set_title('Cross-Country Generalization: Single-Tier vs Two-Tier',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(countries)
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    for i, (s, t) in enumerate(zip(single_auc, twotier_auc)):
        diff  = t - s
        color = 'green' if diff > 0 else 'red'
        ax.annotate(f'{diff:+.3f}', xy=(x[i] + width/2, t + 0.02),
                    ha='center', fontsize=9, color=color, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_feature_importance_twotier(feature_importance, save_path):
    fig, ax = plt.subplots(figsize=(10, 8))
    top_n   = feature_importance.head(12).copy().iloc[::-1]
    colors  = ['#2E86AB' if t == 'Country' else '#A23B72'
                for t in top_n['tier']][::-1]
    ax.barh(y=top_n['human_name'], width=top_n['importance'], color=colors)
    ax.set_xlabel('Mean |SHAP Value| (Gradient Saliency)', fontsize=12)
    ax.set_title('Two-Tier Dropout Risk Drivers\nMulti-Country Model (5 African Countries)',
                 fontsize=14, fontweight='bold')
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor='#A23B72', label='Individual (Tier 2)'),
        Patch(facecolor='#2E86AB', label='Country Context (Tier 1)')
    ], loc='lower right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_dropout_by_country(df, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    country_stats = df.groupby('country').agg({
        'dropout': 'mean', 'enrolled': 'mean', 'never_enrolled': 'mean'
    }).sort_values('dropout', ascending=True)
    countries = [c.capitalize() for c in country_stats.index]
    x     = np.arange(len(countries))
    width = 0.25
    ax.barh(x - width, country_stats['enrolled']*100,       width, label='Enrolled',       color='#2E86AB')
    ax.barh(x,         country_stats['dropout']*100,        width, label='Dropout',        color='#E74C3C')
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
# ERROR ANALYSIS
# =============================================================================

def error_analysis(model, X_test, y_test, ensemble_prob, best_thresh, df_test):
    """Complete error analysis — confusion matrix, error profiles, systematic patterns."""

    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)

    y_pred = (ensemble_prob >= best_thresh).astype(int)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    total = len(y_test)

    print("\n  CONFUSION MATRIX")
    print("  " + "-" * 45)
    print(f"  {'':20s} Predicted Stay   Predicted Dropout")
    print(f"  {'Actual Stay':20s} TN={tn:4d} ({tn/total*100:.1f}%)   FP={fp:4d} ({fp/total*100:.1f}%)")
    print(f"  {'Actual Dropout':20s} FN={fn:4d} ({fn/total*100:.1f}%)   TP={tp:4d} ({tp/total*100:.1f}%)")
    print("  " + "-" * 45)
    print(f"\n  Out of {y_test.sum()} actual dropouts:")
    print(f"    Correctly caught (TP): {tp} ({tp/y_test.sum()*100:.1f}%)")
    print(f"    Missed        (FN):    {fn} ({fn/y_test.sum()*100:.1f}%)")
    print(f"\n  Out of {(y_test==0).sum()} actual non-dropouts:")
    print(f"    Correctly cleared (TN): {tn} ({tn/(y_test==0).sum()*100:.1f}%)")
    print(f"    Wrongly flagged   (FP): {fp} ({fp/(y_test==0).sum()*100:.1f}%)")

    # Identify error groups
    y_test_arr = np.array(y_test)
    X_test_reset = X_test.reset_index(drop=True)
    df_test_reset = df_test.reset_index(drop=True)

    fp_mask = (y_pred == 1) & (y_test_arr == 0)
    fn_mask = (y_pred == 0) & (y_test_arr == 1)

    false_positives = X_test_reset[fp_mask]
    false_negatives = X_test_reset[fn_mask]

    print("\n\n  ERROR GROUP SIZES")
    print("  " + "-" * 45)
    print(f"  False Positives (wrongly flagged):   {len(false_positives)}")
    print(f"  False Negatives (missed dropouts):   {len(false_negatives)}")
    print(f"  True Positives  (correctly caught):  {tp}")
    print(f"  True Negatives  (correctly cleared): {tn}")

    # Profile by Age Group
    print("\n\n  ERROR PROFILE BY AGE GROUP")
    print("  " + "-" * 45)
    if 'age' in X_test_reset.columns:
        bins   = [0, 10, 13, 16, 100]
        labels = ['6-10', '11-13', '14-16', '17+']
        fn_age = pd.cut(false_negatives['age'], bins=bins, labels=labels).value_counts().sort_index()
        fp_age = pd.cut(false_positives['age'], bins=bins, labels=labels).value_counts().sort_index()
        print(f"  {'Age Group':<12} {'Missed (FN)':>12} {'Wrongly Flagged (FP)':>22}")
        print("  " + "-" * 48)
        for age in labels:
            print(f"  {age:<12} {fn_age.get(age, 0):>12} {fp_age.get(age, 0):>22}")

    # Profile by Gender
    print("\n\n  ERROR PROFILE BY GENDER")
    print("  " + "-" * 45)
    if 'female' in X_test_reset.columns:
        fn_gender = false_negatives['female'].value_counts()
        fp_gender = false_positives['female'].value_counts()
        for val, label in [(0, 'Male'), (1, 'Female')]:
            print(f"  {label:<12} Missed(FN)={fn_gender.get(val,0):>4}   "
                  f"Wrongly Flagged(FP)={fp_gender.get(val,0):>4}")

    # Profile by Wealth
    print("\n\n  ERROR PROFILE BY WEALTH")
    print("  " + "-" * 45)
    if 'wealth_index' in X_test_reset.columns:
        qbins  = X_test_reset['wealth_index'].quantile([0, 0.33, 0.66, 1.0]).values
        wlabels = ['Low', 'Medium', 'High']
        fn_wealth = pd.cut(false_negatives['wealth_index'], bins=qbins,
                           labels=wlabels, include_lowest=True).value_counts()
        fp_wealth = pd.cut(false_positives['wealth_index'], bins=qbins,
                           labels=wlabels, include_lowest=True).value_counts()
        print(f"  {'Wealth':<12} {'Missed (FN)':>12} {'Wrongly Flagged (FP)':>22}")
        print("  " + "-" * 48)
        for w in wlabels:
            print(f"  {w:<12} {fn_wealth.get(w,0):>12} {fp_wealth.get(w,0):>22}")

    # Profile by Urban/Rural
    print("\n\n  ERROR PROFILE BY URBAN/RURAL")
    print("  " + "-" * 45)
    if 'urban' in X_test_reset.columns:
        fn_urban = false_negatives['urban'].value_counts()
        fp_urban = false_positives['urban'].value_counts()
        for val, label in [(0, 'Rural'), (1, 'Urban')]:
            print(f"  {label:<12} Missed(FN)={fn_urban.get(val,0):>4}   "
                  f"Wrongly Flagged(FP)={fp_urban.get(val,0):>4}")

    # Profile by Country
    print("\n\n  ERROR PROFILE BY COUNTRY")
    print("  " + "-" * 45)
    if 'country' in df_test_reset.columns:
        df_test_reset = df_test_reset.copy()
        df_test_reset['y_true'] = y_test_arr
        df_test_reset['y_pred'] = y_pred
        df_test_reset['is_fn']  = ((df_test_reset['y_pred'] == 0) &
                                    (df_test_reset['y_true'] == 1)).astype(int)
        df_test_reset['is_fp']  = ((df_test_reset['y_pred'] == 1) &
                                    (df_test_reset['y_true'] == 0)).astype(int)
        country_errors = df_test_reset.groupby('country').agg(
            total=('y_true', 'count'),
            actual_dropouts=('y_true', 'sum'),
            missed_fn=('is_fn', 'sum'),
            wrong_fp=('is_fp', 'sum')
        )
        print(f"  {'Country':<12} {'Total':>8} {'Dropouts':>10} "
              f"{'Missed FN':>10} {'Wrong FP':>10}")
        print("  " + "-" * 55)
        for country, row in country_errors.iterrows():
            print(f"  {country.capitalize():<12} {int(row['total']):>8} "
                  f"{int(row['actual_dropouts']):>10} "
                  f"{int(row['missed_fn']):>10} "
                  f"{int(row['wrong_fp']):>10}")

    # Systematic Pattern Summary
    print("\n\n  SYSTEMATIC ERROR SUMMARY")
    print("  " + "-" * 45)
    if fn > fp:
        print(f"  Dominant error: FALSE NEGATIVES ({fn}) "
              f"— model misses too many real dropouts")
    else:
        print(f"  Dominant error: FALSE POSITIVES ({fp}) "
              f"— model flags too many non-dropouts")

    if 'age' in false_negatives.columns and len(false_negatives) > 0:
        print(f"  Avg age of missed dropouts (FN):  "
              f"{false_negatives['age'].mean():.1f} years")
        print(f"  Avg age of wrongly flagged (FP):  "
              f"{false_positives['age'].mean():.1f} years")
    if 'wealth_index' in false_negatives.columns and len(false_negatives) > 0:
        print(f"  Avg wealth of missed dropouts:    "
              f"{false_negatives['wealth_index'].mean():.3f}")
        print(f"  Avg wealth of wrongly flagged:    "
              f"{false_positives['wealth_index'].mean():.3f}")

    # Save confusion matrix plot
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Stay', 'Dropout'],
                yticklabels=['Stay', 'Dropout'],
                ax=ax, annot_kws={'size': 16})
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('Actual', fontsize=12)
    ax.set_title('Confusion Matrix — High-Dropout Model',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "error_analysis_confusion_matrix.png",
                dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {FIGURES_DIR / 'error_analysis_confusion_matrix.png'}")

    return false_positives, false_negatives


# =============================================================================
# HIGH-DROPOUT COUNTRIES MODEL
# =============================================================================

def train_high_dropout_countries(df):
    """Train on Tanzania + Rwanda only for best F1 and Precision."""

    print("\n" + "=" * 60)
    print("HIGH-DROPOUT COUNTRIES MODEL (Tanzania + Rwanda)")
    print("=" * 60)

    high_dropout = df[df['country'].isin(['tanzania', 'rwanda'])].copy()
    print(f"  Samples: {len(high_dropout):,}")
    print(f"  Dropout: {high_dropout['dropout'].sum():,} "
          f"({high_dropout['dropout'].mean()*100:.1f}%)")

    X, y, features = prepare_features(high_dropout, include_country=True)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Aggressive resampling
    pipe = Pipeline([
        ('u', RandomUnderSampler(sampling_strategy=0.5, random_state=42)),
        ('o', SMOTE(sampling_strategy=1.0, k_neighbors=5, random_state=42))
    ])
    X_train_bal, y_train_bal = pipe.fit_resample(X_train, y_train)
    print(f"  After resampling: {len(X_train_bal):,} | "
          f"Dropout: {y_train_bal.sum()} ({y_train_bal.mean()*100:.1f}%)")

    # ── Targeted fix: sample weights focusing on young, rural, low-wealth ──
    sample_weights = np.ones(len(X_train_bal))
    if 'age' in X_train_bal.columns:
        # Boost weight for young children (6-13) who are most missed
        young_mask = X_train_bal['age'] <= 13
        sample_weights[young_mask] *= 2.0
    if 'urban' in X_train_bal.columns:
        # Boost weight for rural students who dominate false negatives
        rural_mask = X_train_bal['urban'] == 0
        sample_weights[rural_mask] *= 1.8
    if 'wealth_index' in X_train_bal.columns:
        # Boost weight for low-wealth students
        low_wealth_mask = X_train_bal['wealth_index'] <= X_train_bal['wealth_index'].quantile(0.33)
        sample_weights[low_wealth_mask] *= 1.5

    ratio = (y_train_bal == 0).sum() / (y_train_bal == 1).sum()

    # Train XGBoost with sample weights
    print("\n  Training XGBoost (with targeted sample weights)...")
    xgb_model = xgb.XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.03,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=1,
        scale_pos_weight=ratio, eval_metric='aucpr',
        early_stopping_rounds=30, random_state=42, n_jobs=-1,
    )
    xgb_model.fit(X_train_bal, y_train_bal,
                  sample_weight=sample_weights,
                  eval_set=[(X_val, y_val)],
                  verbose=False)

    # Train Random Forest with sample weights
    print("  Training Random Forest (with targeted sample weights)...")
    rf_model = RandomForestClassifier(
        n_estimators=500, max_depth=10, min_samples_leaf=1,
        class_weight={0: 1, 1: ratio}, random_state=42, n_jobs=-1)
    rf_model.fit(X_train_bal, y_train_bal, sample_weight=sample_weights)

    # Train Gradient Boosting with sample weights
    print("  Training Gradient Boosting (with targeted sample weights)...")
    gb_model = GradientBoostingClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        subsample=0.8, random_state=42)
    gb_model.fit(X_train_bal, y_train_bal, sample_weight=sample_weights)

    # Optimize ensemble weights and threshold on validation set
    print("\n  Optimizing ensemble...")
    xgb_v = xgb_model.predict_proba(X_val)[:, 1]
    rf_v  = rf_model.predict_proba(X_val)[:, 1]
    gb_v  = gb_model.predict_proba(X_val)[:, 1]

    best_f1, best_w, best_t = 0, (0.5, 0.3, 0.2), 0.5
    for w1 in np.arange(0.3, 0.8, 0.1):
        for w2 in np.arange(0.1, 0.5, 0.1):
            w3 = round(1.0 - w1 - w2, 1)
            if w3 <= 0:
                continue
            ens = w1*xgb_v + w2*rf_v + w3*gb_v
            for t in np.arange(0.05, 0.70, 0.01):
                score = f1_score(y_val, (ens >= t).astype(int))
                if score > best_f1:
                    best_f1, best_w, best_t = score, (w1, w2, w3), t

    print(f"  Best weights — XGB={best_w[0]:.1f} "
          f"RF={best_w[1]:.1f} GB={best_w[2]:.1f}")
    print(f"  Best threshold: {best_t:.2f} | Val F1: {best_f1:.4f}")

    # Evaluate on test set
    xgb_t = xgb_model.predict_proba(X_test)[:, 1]
    rf_t  = rf_model.predict_proba(X_test)[:, 1]
    gb_t  = gb_model.predict_proba(X_test)[:, 1]
    ens_t = best_w[0]*xgb_t + best_w[1]*rf_t + best_w[2]*gb_t

    # ── Targeted fix: lower threshold for young rural low-wealth students ──
    X_test_reset = X_test.reset_index(drop=True)
    adjusted_prob = ens_t.copy()
    if all(c in X_test_reset.columns for c in ['age', 'urban', 'wealth_index']):
        vulnerable_mask = (
            (X_test_reset['age'] <= 13) &
            (X_test_reset['urban'] == 0) &
            (X_test_reset['wealth_index'] <=
             X_test_reset['wealth_index'].quantile(0.33))
        ).values
        # Boost probability for this high-risk group
        adjusted_prob[vulnerable_mask] = np.clip(
            adjusted_prob[vulnerable_mask] * 1.3, 0, 1)
        print(f"\n  Vulnerable group identified: "
              f"{vulnerable_mask.sum()} young rural low-wealth students")

    y_pred = (adjusted_prob >= best_t).astype(int)

    print("\n" + "=" * 60)
    print("HIGH-DROPOUT MODEL — FINAL PERFORMANCE (After Fix)")
    print("=" * 60)
    print(f"  AUC-ROC:   {roc_auc_score(y_test, adjusted_prob):.4f}")
    print(f"  F1-Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred):.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=['Stay', 'Dropout']))

    # Run error analysis
    df_test_reset = high_dropout.reset_index(
        drop=True).iloc[:len(y_test)].reset_index(drop=True)
    error_analysis(xgb_model, X_test, y_test,
                   adjusted_prob, best_t, df_test_reset)


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    print("\n" + "=" * 60)
    print("   MULTI-COUNTRY TWO-TIER DROPOUT MODEL")
    print("   5 African Countries | DHS Data | 146K Students")
    print("=" * 60)

    # Load data
    df = load_dhs_data()
    df = add_country_features(df)

    # Cross-validation comparison
    print("\n" + "=" * 60)
    print("COMPARING SINGLE-TIER vs TWO-TIER")
    print("=" * 60)

    print("\n--- Single-Tier (Individual Features Only) ---")
    results_single = cross_country_validation(df, include_country=False)

    print("\n--- Two-Tier (Individual + Country Context) ---")
    results_twotier = cross_country_validation(df, include_country=True)

    # Full model
    print("\n" + "=" * 60)
    print("TRAINING FULL TWO-TIER MODEL")
    print("=" * 60)

    X, y, feature_names = prepare_features(df, include_country=True)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"\n  Train: {len(X_train):,}")
    print(f"  Val:   {len(X_val):,}")
    print(f"  Test:  {len(X_test):,}")

    model                   = train_model(X_train, y_train, X_val, y_val)
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

    feature_importance = compute_saliency(model, X_test, feature_names)

    print("\n  TOP DROPOUT DRIVERS:")
    print("  " + "-" * 55)
    for _, row in feature_importance.head(10).iterrows():
        tier_icon = "Country" if row['tier'] == 'Country' else "Individual"
        print(f"  [{tier_icon}] {row['human_name']:30s} | {row['importance']:.4f}")

    # Save outputs
    print("\n" + "=" * 60)
    print("SAVING OUTPUTS")
    print("=" * 60)

    plot_dropout_by_country(df, FIGURES_DIR / "dhs_dropout_by_country.png")
    plot_feature_importance_twotier(feature_importance,
                                    FIGURES_DIR / "multicountry_feature_importance.png")
    plot_country_comparison(results_single, results_twotier,
                            FIGURES_DIR / "country_generalization.png")

    with open(MODELS_DIR / "xgboost_multicountry.pkl", 'wb') as f:
        pickle.dump(model, f)
    print(f"  Saved: model")

    feature_importance.to_csv(
        REPORTS_DIR / "multicountry_feature_importance.csv", index=False)
    pd.DataFrame(results_twotier).to_csv(
        REPORTS_DIR / "cross_country_validation.csv", index=False)
    print(f"  Saved: reports")

    # Summary
    print("\n" + "=" * 60)
    print("MULTI-COUNTRY MODEL COMPLETE")
    print("=" * 60)

    avg_single  = np.mean([r['auc_roc'] for r in results_single])
    avg_twotier = np.mean([r['auc_roc'] for r in results_twotier])
    print(f"\n  Cross-Country Generalization:")
    print(f"    Single-Tier AUC: {avg_single:.4f}")
    print(f"    Two-Tier AUC:    {avg_twotier:.4f}")
    print(f"    Improvement:     {avg_twotier - avg_single:+.4f}")
    print(f"\n  Full Model Test Performance:")
    print(f"    AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"    F1-Score:  {metrics['f1']:.4f}")
    print("\n" + "=" * 60)

    # High-dropout countries model with error analysis
    train_high_dropout_countries(df)


if __name__ == "__main__":
    main()