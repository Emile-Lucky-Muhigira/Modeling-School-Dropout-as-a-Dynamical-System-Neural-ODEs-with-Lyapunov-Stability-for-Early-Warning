PROJECT TITLE: Modeling School Dropout as a Dynamical System: Neural ODEs with Lyapunov Stability for Early Warning in East Africa

AUTHOR: Emile Lucky Muhigira
AFFILIATION: Carnegie Mellon University Africa
DATE: May 2026

================================================================================
I. PROJECT OVERVIEW
================================================================================
This research presents a novel early warning system for school dropout prediction in Sub-Saharan Africa. Unlike conventional models that treat dropout as a static event, this system models educational disengagement as a continuous dynamical process. 

The project leverages a hybrid approach:
1.  A Two-Tier Ensemble Model for broad screening across multi-country contexts.
2.  A Neural Ordinary Differential Equation (Neural ODE) component for high-fidelity trajectory tracking and prioritization[cite: 5].

The system was validated on a combined dataset of 158,684 students across five East African countries (Ethiopia, Kenya, Rwanda, Tanzania, and Uganda), achieving state-of-the-art performance in identifying at-risk populations within imbalanced datasets[cite: 5].

================================================================================
II. KEY CONTRIBUTIONS & INNOVATIONS
================================================================================
* DYNAMICAL MODELING: Introduced Neural ODEs to handle irregularly sampled longitudinal survey data, enabling the modeling of student disengagement as a continuous trajectory[cite: 5].
* LYAPUNOV STABILITY ANALYSIS: Integrated control theory principles to produce a "Trajectory Signal"[cite: 5]. This allows practitioners to distinguish between students who are stabilizing and those whose risk is actively worsening[cite: 5].
* TWO-TIER ARCHITECTURE: Developed a system that encodes country-level systemic awareness (national enrollment and dropout trends) combined with individual-level feature prediction to improve regional generalization[cite: 5].
* SYSTEMATIC ERROR ANALYSIS: Conducted a rigorous post-hoc analysis identifying Rural Bias (78.8% of missed cases) and age-specific failure modes, providing a concrete roadmap for future data-driven educational policy[cite: 5].

================================================================================
III. TECHNICAL SPECIFICATIONS
================================================================================
* ALGORITHMS: Neural ODEs (torchdiffeq), XGBoost, Random Forest, Gradient Boosting Ensembles[cite: 5].
* MATHEMATICAL FRAMEWORK: Lyapunov Stability Theory, Continuous-depth Neural Networks[cite: 5].
* INTERPRETABILITY: SHAP (SHapley Additive exPlanations) for identifying multi-country risk drivers[cite: 5].
* DATASETS: 
    - Young Lives Ethiopia (Longitudinal Cohort Study, n=11,996)[cite: 5]
    - Demographic and Health Surveys (DHS) Multi-Country (Cross-sectional, n=146,688)[cite: 5]
* METRICS: Prioritized F1-score for the minority dropout class, AUC-ROC, Recall, and Precision[cite: 5].

================================================================================
IV. KEY PERFORMANCE RESULTS
================================================================================
* High-Dropout Contexts (Tanzania/Rwanda): Achieved an F1 score of 0.6804 and a Recall of 0.7044 for the dropout class[cite: 5].
* Cross-Country Generalization: Maintained a mean AUC of 0.861 across five leave-one-country-out validation experiments[cite: 5].
* Actionable Insights: Students in the "ambiguous risk" band (50-70%) with worsening trajectories were found to be nearly twice as likely (31% vs 18%) to drop out compared to those stabilizing[cite: 5].

================================================================================
V. PRACTICAL IMPACT & APPLICATIONS
================================================================================
The system provides a two-step operational roadmap for educational administrators:
1.  SCREENING: Rapid identification of at-risk populations using the ensemble tier[cite: 5].
2.  TRIAGE: Using the Lyapunov trajectory signal to prioritize limited counseling resources for students on a deteriorating path[cite: 5].

================================================================================
VI. RELEVANT PUBLICATIONS & REFERENCES
================================================================================
Full documentation, methodology, and comprehensive error analysis are available in the technical report: "Modeling School Dropout as a Dynamical System: Neural ODEs with Lyapunov Stability for Early Warning in East Africa" by Emile Lucky Muhigira[cite: 5].