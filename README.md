# ADHD & Gender Prediction from fMRI and Clinical Data 🧠


## 🚀 Objective

Predict:
- `ADHD_OUTCOME`: Whether the subject is diagnosed with ADHD (0 or 1).
- `SEX_F`: Whether the subject is biologically female (0 or 1).

Based on:
- **fMRI functional connectome matrices** (19900 features from 200x200 correlation matrix)
- **Categorical metadata** (age group, handedness, etc.)

---

---

## 📊 Models Used

- **RandomForestClassifier** with SMOTE balancing
- **LightGBMRegressor** with threshold optimization (for SEX_F)
- **Cross-validation** with StratifiedKFold (5 folds)

---

## 🛠️ Key Features

- ✅ Custom meta-features extracted from fMRI data (mean, std, skewness, kurtosis)
- ✅ SMOTE balancing for ADHD & SEX_F
- ✅ Hyperparameter-free baseline for fast experimentation
- ✅ Threshold tuning for regression-based classification (SEX_F)
- ✅ Submission file auto-generation

---

## 📈 Results

| Target        | Best CV F1 Score |
|---------------|------------------|
| ADHD_OUTCOME  | 0.797            |
| SEX_F         | 0.516 (with LightGBM + meta features) |

---
