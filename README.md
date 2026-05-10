# Clinical Trial Early Warning System

## Overview

This project builds an early warning system (with a machine learning model) to predict clinical trial success or failure using structured data from ClinicalTrials.gov. By generating time-based snapshots at 3, 6, and 12 months, the model identifies high-risk trials early in their lifecycle and supports proactive decision-making in clinical development, that leads to cost-savings and improved efficiency.

The system combines machine learning classification, survival analysis, and explainability techniques to evaluate trial feasibility, operational risk, and outcome likelihood.

## Problem Statement

Clinical trials are expensive, time-intensive, and prone to late-stage failure. Many failures occur after substantial financial and operational investment has already been made.

The goal of this project is to:
- Predict trial success vs. failure as early as possible
- Identify the strongest drivers of trial outcomes
- Support resource allocation and intervention decisions using data-driven insights
- Measure project success using key performance metrics like ROC-AUC, average precision, accuracy, recall, F1-score

## Dataset

### Source
- ClinicalTrials.gov API (v2)
  - https://clinicaltrials.gov/api/v2/studies

### Scope
- Cancer-related studies
- Phase II and Phase III trials
- Trial statuses:
  - Completed → labeled as Success
  - Terminated / Withdrawn → labeled as Failure

### Final Dataset
- 495 trials
- 1,455 total snapshots
- Snapshot windows:
  - 3 months
  - 6 months
  - 12 months

- Key variables
  - Trial Identification: NCTId, BriefTitle, Condition
  - Trial Design: Phase, StudyType, InterventionType, ArmGroup
  - Enrollment: EnrollmentCount, EnrollmentType
  - Status: OverallStatus, StatusHistory
  - Timeline: StartDate, CompletionDate


### Tech Stack
- Languages: Python
- Libraries: pandas, lumpy, requests, tqdm, scikit-learn, xgboost, lifelines, shap, matplotlib
- Tools: Jupyter Notebook

### Methodology

1. Data Cleaning
Raw clinical trial records were collected from the ClinicalTrials.gov API and required significant preprocessing before modeling.

  - Filtered studies to include only: Cancer-related trials, Phase II and Phase III studies, Final statuses of Completed, Terminated, or Withdrawn
  - Removed trials with missing or invalid start dates
  - Excluded trials that ended before the selected snapshot windows (3, 6, 12 months)
  - Standardized date formats across multiple inconsistent formats
  - Handled missing numerical values using median imputation
  - Removed low-variance and highly correlated variables for survival modeling stability

This ensured consistent labels and realistic early-stage prediction scenarios.

2. EDA
EDA was performed to understand class balance, feature distributions, and trial behavior across time windows.

Key Findings
  - Success and failure labels remained relatively balanced across snapshots
  - Larger enrollment targets were associated with higher failure risk
  - Enrollment behavior emerged as one of the strongest predictive signals

3. Feature Engineering
Features were generated using only data available at 3, 6, and 12 months.

Trial Design Features
  - Phase number
  - Sponsor category
  - Number of trial arms
  - Number of primary outcomes
Intervention type indicators
  - Operational Features
  - Enrollment velocity
  - Number of trial sites
  - Number of countries
  - Recruitment status indicators
  - Suspension history
  - Number of status changes
Complexity Features
  - Planned enrollment size (log transformed)
  - Eligibility criteria length
  - Protocol amendment count
  - Multinational trial indicator

4. Modeling
  - XGBoost Classifier
Why XGBoost
  - Strong performance on structured tabular data
  - Handles nonlinear interactions effectively
  - Robust with missing values
  - Supports explainability through SHAP values
Training Strategy
  - Time-based train/test split using trial start year
  - 5-fold stratified cross-validation
  - Standardization and median imputation pipeline
  - Class imbalance adjustment using weighted learning

  - Cox Proportional Hazards Model
Why Cox
  - Used to identify time-dependent failure risks and understand how specific variables influence trial duration and termination likelihood.
This improved interpretability beyond binary classification alone.


5. Evaluation
Model performance was evaluated across all snapshot windows to measure early prediction effectiveness.

Classification Metrics
  - ROC-AUC
  - Average Precision
  - Accuracy
  - Precision
  - Recall
  - F1-score

Explainability
  - SHAP summary plots were used to identify the most important predictive features

## Results

| Snapshot Window | AUC | Avg Precision | Accuracy |
|---|---:|---:|---:|
| 3 Months | 0.876 | 0.909 | 82% |
| 6 Months | 0.875 | 0.907 | 84% |
| 12 Months | 0.872 | 0.916 | 85% |

- Key Takeaways:
  - Early-stage prediction is highly effective, with the model achieving strong performance (ROC-AUC > 0.87) as early as 3 months into trial execution, enabling proactive intervention rather than late-stage reaction.
  - Enrollment feasibility is the strongest predictor of trial success, with planned enrollment size and enrollment velocity consistently ranking as the most important features across all snapshot windows.
  - Trial complexity increases failure risk, as studies with more trial arms, more primary outcomes, longer eligibility criteria, and broader multinational operations were more likely to terminate or withdraw.
  - Dynamic operational behavior is more predictive than static trial design alone, showing that how a trial progresses early is often more important than how it was originally planned.
  - Consistent performance across 3-, 6-, and 12-month snapshots validates the robustness of the modeling framework and supports real-world deployment as an early warning decision support system.
  - SHAP explainability and survival analysis improved model transparency, helping identify actionable risk drivers and making predictions more interpretable for stakeholders in clinical development.

## How to Run

```bash
# Clone the repo
git clone https://github.com/AaronTsuiCareers/ClinicalTrialPred.git

# Navigate
cd ClinicalTrialPred

# Install dependencies
pip install -r requirements.txt

# Run
jupyter notebook
```
## Project Structure

```text
ClinicalTrialPred/
│
├── main.ipynb    
├── clinical_trials_raw.csv       
├── early_warning_snapshots.csv       
├── results.txt
├── requirements.txt                                        
└── README.md     
```

---

## Author

### Aaron Tsui

Master of Science in Data Science  
Northwestern University 

Specializing in machine learning, deep learning, and applied AI systems for healthcare and large-scale predictive modeling.

- Email: aaron.tsui.careers@gmail.com  
- LinkedIn: https://www.linkedin.com/in/aaron-tsui/

