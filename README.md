# Adult Census Income - Fairness in Machine Learning

## Overview
This project applies fairness analysis on the Adult Census Income 
dataset to detect and mitigate bias in income prediction models 
across demographic groups including gender, race and age.

## Group
Group 1 - Bora Malaj
Assigned Task: Fairness Analysis and Bias Mitigation

## Dataset
- Source: Kaggle - Adult Census Income
- Samples: 32561 records
- Features: 15 features including protected attributes
- Target: income (<=50K or >50K)

## Protected Attributes
| Attribute | Values |
|-----------|--------|
| sex | Male, Female |
| race | White, Black, Asian, Other |
| age | Grouped into <25, 25-35, 35-45, 45-55, 55-65, 65+ |
| native.country | 40+ countries |

## Steps Covered
1. EDA - Count by gender, race, age group
2. Identify underrepresented groups
3. Train Logistic Regression and Decision Tree
4. Fairness analysis - precision, recall, F1 by group
5. Mitigation - reweighting underrepresented groups
6. Compare fairness metrics before and after mitigation

## Libraries Used
- pandas, numpy, matplotlib, seaborn, scikit-learn

## Author
Bora Malaj
