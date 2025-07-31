# Insights on Model Performance

## 1. Random Forest on Breast Cancer Dataset
- **Accuracy:** High
- **Strengths:** Performs well on structured/tabular data.
- **Confusion Matrix:** Very low false positives.
- **ROC AUC:** Near 1.0

## 2. Logistic Regression on NLP (20 Newsgroups)
- **Accuracy:** Moderate
- **Insights:** TF-IDF + Logistic Regression is simple but effective.
- **Confusion Matrix:** Some confusion between classes.
- **ROC AUC:** Fairly good

## 3. ResNet50 on CIFAR-10 (Image Classification)
- **Accuracy:** Moderate due to limited training (subset used).
- **Notes:** Very deep model; potential increases with longer training.
- **Visual Data:** Performs better with more epochs/data.

## Overall Reflection
- Different domains require different model types.
- Evaluation through ROC & confusion matrix helps understand strengths.
- Future: Try ensemble models or transformers for better generalization.
