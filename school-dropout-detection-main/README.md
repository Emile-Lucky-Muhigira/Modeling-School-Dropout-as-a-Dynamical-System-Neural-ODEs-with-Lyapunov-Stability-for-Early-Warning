School Dropout Detection

Explainable Early Warning System for School Dropout Prediction in African Education Systems



Authors: Emile Lucky Muhigira, Martin Mulang'





Project Overview

This project builds an explainable machine learning system that:



Predicts which students are at risk of dropping out

Explains why using gradient-based saliency mapping

Tracks risk trajectories using Lyapunov stability analysis

Key Innovation

Most dropout models just say "High Risk." Our system says:



"Student #247 has 52.3% dropout risk. Primary driver: Peer Stability Index collapsed by 0.4 points (∇ = −0.41). Recommended intervention: peer mentoring program."



Data Sources

Young Lives Ethiopia (Primary Dataset)

Source: UK Data Service (Study 7483)

Coverage: 2,999 children tracked across 5 rounds (2002-2016)

Observations: 14,995 child-round records

Variables: 214 features including education, household, health

Dropout Rate: \~20.5% (614 children dropped out)

Kaggle Education in Africa (Country-Level)

Coverage: 54 African countries, 2010-2023

Observations: 756 country-year records

Variables: 60+ indicators on enrollment, attendance, teachers, expenditure

Model Architecture

┌─────────────────────────────────────────────────────────────┐

│                    TWO-TIER ARCHITECTURE                     │

├─────────────────────────────────────────────────────────────┤

│                                                              │

│  TIER 1: Country-Level (Kaggle)      TIER 2: Student-Level  │

│  ┌─────────────────────────┐         ┌───────────────────┐  │

│  │  XGBoost Classifier     │         │  Neural ODE       │  │

│  │  - Country risk score   │         │  - Lyapunov V(x)  │  │

│  │  - Macro indicators     │         │  - dV/dt tracking │  │

│  └─────────────────────────┘         └───────────────────┘  │

│              │                                │              │

│              └──────────┬─────────────────────┘              │

│                         ▼                                    │

│              ┌─────────────────────┐                        │

│              │  Combined Risk:     │                        │

│              │  α·V(x) + (1-α)·ŷ   │                        │

│              └─────────────────────┘                        │

│                         │                                    │

│                         ▼                                    │

│              ┌─────────────────────┐                        │

│              │  Gradient Saliency  │                        │

│              │  ∇V → Top Drivers   │                        │

│              └─────────────────────┘                        │

│                         │                                    │

│                         ▼                                    │

│              ┌─────────────────────┐                        │

│              │  Actionable Alert   │                        │

│              │  "Risk: 52.3%       │                        │

│              │   Driver: Peer      │                        │

│              │   Action: Mentor"   │                        │

│              └─────────────────────┘                        │

└─────────────────────────────────────────────────────────────┘

Quick Start

1\. Setup Environment

cd school-dropout-detection

python -m venv venv

venv\\Scripts\\activate  # Windows

\# source venv/bin/activate  # Mac/Linux



pip install -r requirements.txt

2\. Verify Data

python src/config.py

3\. Run Training

python src/train.py

Project Structure

school-dropout-detection/

├── data/

│   ├── raw/

│   │   ├── young\_lives/     # Young Lives .dta files

│   │   └── kaggle/          # Education in Africa CSVs

│   └── processed/

│       ├── ethiopia\_dropout\_panel.csv      # Ready for modeling

│       └── ethiopia\_constructed\_full.csv   # Full 214 variables

├── src/

│   ├── config.py            # Configuration

│   ├── data\_pipeline.py     # Data loading

│   ├── xgboost\_model.py     # XGBoost classifier

│   ├── neural\_ode.py        # Neural ODE with Lyapunov

│   ├── saliency.py          # Gradient saliency mapping

│   └── train.py             # Training orchestration

├── notebooks/

│   └── exploration.ipynb    # Data exploration

├── outputs/

│   ├── models/              # Saved models

│   ├── figures/             # Plots

│   └── reports/             # Generated reports

├── paper/

│   └── main.tex             # Paper draft

├── results/

│   └── mlruns/              # MLflow tracking

├── requirements.txt

└── README.md

Key Variables

Target Variable

dropout\_next\_round: 1 if child drops out in next survey round, 0 otherwise

Feature Categories

Category	Variables	Description

Education	enrol, engrade, hghgrade, preprim	Enrollment, grade level

Household	wi\_new, hq\_new, hhsize, ownhouse	Wealth, housing quality

Parental	momedu, dadedu, momlive, dadlive	Parent education, presence

Child	chsex, agemon, stunting, underweight	Demographics, health

Shocks	shecon\*, shenv\*, shfam\*	Economic, environmental, family shocks

Location	region, typesite, commid	Urban/rural, region

Evaluation Metrics

AUC-ROC: Primary metric for imbalanced classification

F1-Score: Balance of precision and recall

Precision@K: Precision for top-K highest risk students

Lead Time: How early can we detect dropout risk?

Feature Importance Alignment: Do top features match domain knowledge?

License

This project uses data from:



UK Data Service (Young Lives) - requires registration

Kaggle (Education in Africa) - open dataset

Contact

Emile Lucky Muhigira - emuhigir@andrew.cmu.edu
Martin Mulang' - mmulang@andrew.cmu.edu

Carnegie Mellon University Africa

