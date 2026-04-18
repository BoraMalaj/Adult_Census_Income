#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('adult.csv')

print("Dataset loaded successfully!")
print("Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
df.head()


# In[3]:


print("Basic Information:")
print("=" * 50)
print(f"Total Records : {df.shape[0]}")
print(f"Total Columns : {df.shape[1]}")
print(f"\nData Types:")
print(df.dtypes)
print(f"\nMissing Values:")
print(df.isnull().sum())
print(f"\nStatistical Summary:")
print(df.describe())


# In[5]:


print("Count by Gender:")
print(df['sex'].value_counts())
print("\nPercentage:")
print((df['sex'].value_counts() / len(df) * 100).round(2))

plt.figure(figsize=(8, 5))
df['sex'].value_counts().plot(kind='bar', 
                               color=['steelblue', 'orange'],
                               edgecolor='black')
plt.title('Count by Gender')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# In[7]:


print("Count by Race:")
print(df['race'].value_counts())
print("\nPercentage:")
print((df['race'].value_counts() / len(df) * 100).round(2))

plt.figure(figsize=(10, 5))
df['race'].value_counts().plot(kind='bar',
                                color='steelblue',
                                edgecolor='black')
plt.title('Count by Race')
plt.xlabel('Race')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# In[9]:


# Create age groups
bins = [0, 25, 35, 45, 55, 65, 100]
labels = ['<25', '25-35', '35-45', '45-55', '55-65', '65+']
df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)

print("Count by Age Group:")
print(df['age_group'].value_counts().sort_index())
print("\nPercentage:")
print((df['age_group'].value_counts().sort_index() / len(df) * 100).round(2))

plt.figure(figsize=(10, 5))
df['age_group'].value_counts().sort_index().plot(
    kind='bar', color='steelblue', edgecolor='black')
plt.title('Count by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# In[11]:


fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Gender vs Income
gender_income = df.groupby(['sex', 'income']).size().unstack()
gender_income.plot(kind='bar', ax=axes[0],
                   color=['steelblue', 'orange'],
                   edgecolor='black', alpha=0.8)
axes[0].set_title('Income Distribution by Gender')
axes[0].set_xlabel('Gender')
axes[0].set_ylabel('Count')
axes[0].tick_params(axis='x', rotation=0)
axes[0].legend(['<=50K', '>50K'])

# Race vs Income
race_income = df.groupby(['race', 'income']).size().unstack()
race_income.plot(kind='bar', ax=axes[1],
                 color=['steelblue', 'orange'],
                 edgecolor='black', alpha=0.8)
axes[1].set_title('Income Distribution by Race')
axes[1].set_xlabel('Race')
axes[1].set_ylabel('Count')
axes[1].tick_params(axis='x', rotation=45)
axes[1].legend(['<=50K', '>50K'])

plt.suptitle('Income Distribution by Protected Attributes', fontsize=14)
plt.tight_layout()
plt.show()


# In[13]:


age_income = df.groupby(['age_group', 'income']).size().unstack()
age_income.plot(kind='bar', figsize=(12, 5),
                color=['steelblue', 'orange'],
                edgecolor='black', alpha=0.8)
plt.title('Income Distribution by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(['<=50K', '>50K'])
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# In[17]:


print("Underrepresented Groups Analysis:")
print("=" * 60)

# Gender representation
print("\n Gender Representation:")
gender_pct = df['sex'].value_counts(normalize=True) * 100
for gender, pct in gender_pct.items():
    status = " Underrepresented" if pct < 30 else " Represented"
    print(f"  {gender:<10} : {pct:.1f}% {status}")

# Race representation
print("\n Race Representation:")
race_pct = df['race'].value_counts(normalize=True) * 100
for race, pct in race_pct.items():
    status = " Underrepresented" if pct < 10 else " Represented"
    print(f"  {race:<30} : {pct:.1f}% {status}")

# Age group representation
print("\n Age Group Representation:")
age_pct = df['age_group'].value_counts(normalize=True).sort_index() * 100
for age, pct in age_pct.items():
    status = " Underrepresented" if pct < 10 else " Represented"
    print(f"  {str(age):<10} : {pct:.1f}% {status}")


# In[19]:


print("Top 10 Native Countries:")
print(df['native.country'].value_counts().head(10))

plt.figure(figsize=(12, 5))
df['native.country'].value_counts().head(15).plot(
    kind='bar', color='steelblue', edgecolor='black')
plt.title('Top 15 Native Countries')
plt.xlabel('Country')
plt.ylabel('Count')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# In[21]:


plt.figure(figsize=(12, 8))
sns.heatmap(df.select_dtypes(include=['int64', 'float64']).corr(),
            annot=True, fmt='.2f', cmap='coolwarm',
            linewidths=0.5, vmin=-1, vmax=1)
plt.title('Correlation Heatmap - Adult Census Income', fontsize=14)
plt.tight_layout()
plt.show()


# In[25]:


from sklearn.preprocessing import LabelEncoder

# Drop missing values marked as '?'
df_model = df.replace('?', np.nan).dropna()
print("Shape after dropping missing values:", df_model.shape)

# Encode all categorical columns
le = LabelEncoder()
cat_cols = df_model.select_dtypes(include=['str']).columns.tolist()

for col in cat_cols:
    df_model[col] = le.fit_transform(df_model[col])

print("\nEncoding done!")
print(df_model.dtypes)


# In[27]:


from sklearn.model_selection import train_test_split

# Features and target
X = df_model.drop(['income', 'age_group'], axis=1)
y = df_model['income']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

print("Data split successfully!")
print(f"Train size : {X_train.shape}")
print(f"Test size  : {X_test.shape}")
print(f"\nClass Distribution:")
print(f"<=50K : {sum(y==0)}")
print(f">50K  : {sum(y==1)}")


# In[48]:


from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Scale the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression on scaled data
lr = LogisticRegression(max_iter=10000, random_state=42)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

print("Logistic Regression Results:")
print("=" * 50)
print(f"Accuracy: {round(accuracy_score(y_test, y_pred_lr), 2)}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr,
      target_names=['<=50K', '>50K']))


# In[50]:


from sklearn.tree import DecisionTreeClassifier

dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

print("Decision Tree Results:")
print("=" * 50)
print(f"Accuracy: {round(accuracy_score(y_test, y_pred_dt), 2)}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_dt,
      target_names=['<=50K', '>50K']))


# In[52]:


from sklearn.metrics import confusion_matrix

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Logistic Regression
cm_lr = confusion_matrix(y_test, y_pred_lr)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['<=50K', '>50K'],
            yticklabels=['<=50K', '>50K'])
axes[0].set_title('Confusion Matrix - Logistic Regression')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# Decision Tree
cm_dt = confusion_matrix(y_test, y_pred_dt)
sns.heatmap(cm_dt, annot=True, fmt='d', cmap='Greens', ax=axes[1],
            xticklabels=['<=50K', '>50K'],
            yticklabels=['<=50K', '>50K'])
axes[1].set_title('Confusion Matrix - Decision Tree')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.suptitle('Confusion Matrices - No Fairness Considerations', fontsize=14)
plt.tight_layout()
plt.show()


# In[54]:


from sklearn.metrics import roc_curve, roc_auc_score

y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_dt, tpr_dt, _ = roc_curve(y_test, y_prob_dt)

auc_lr = roc_auc_score(y_test, y_prob_lr)
auc_dt = roc_auc_score(y_test, y_prob_dt)

plt.figure(figsize=(10, 6))
plt.plot(fpr_lr, tpr_lr, color='steelblue', linewidth=2,
         label=f'Logistic Regression (AUC = {auc_lr:.2f})')
plt.plot(fpr_dt, tpr_dt, color='green', linewidth=2,
         label=f'Decision Tree (AUC = {auc_dt:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--',
         label='Random Classifier')
plt.title('ROC Curve Comparison')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()


# In[56]:


from sklearn.metrics import precision_score, recall_score, f1_score

results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Decision Tree'],
    'Accuracy': [
        round(accuracy_score(y_test, y_pred_lr), 2),
        round(accuracy_score(y_test, y_pred_dt), 2)
    ],
    'Precision': [
        round(precision_score(y_test, y_pred_lr), 2),
        round(precision_score(y_test, y_pred_dt), 2)
    ],
    'Recall': [
        round(recall_score(y_test, y_pred_lr), 2),
        round(recall_score(y_test, y_pred_dt), 2)
    ],
    'F1 Score': [
        round(f1_score(y_test, y_pred_lr), 2),
        round(f1_score(y_test, y_pred_dt), 2)
    ],
    'AUC': [
        round(auc_lr, 2),
        round(auc_dt, 2)
    ]
})

print("Model Comparison - No Fairness Considerations:")
print("=" * 60)
print(results.to_string(index=False))


# In[58]:


# Decision Tree feature importance
importances = pd.Series(dt.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

plt.figure(figsize=(12, 6))
importances.head(10).plot(kind='bar', color='steelblue', edgecolor='black')
plt.title('Top 10 Feature Importances - Decision Tree')
plt.xlabel('Features')
plt.ylabel('Importance Score')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

print("\nTop 5 Most Important Features:")
print(importances.head())


# In[60]:


# Add predictions to test set
X_test_df = X_test.copy()
X_test_df['actual'] = y_test.values
X_test_df['predicted_lr'] = y_pred_lr
X_test_df['predicted_dt'] = y_pred_dt

print("Predictions added!")
print(X_test_df.head())


# In[68]:


from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

print("Fairness Analysis by Gender:")
print("=" * 60)

for gender_code, gender_name in [(0, 'Female'), (1, 'Male')]:
    mask = X_test_df['sex'] == gender_code
    actual = X_test_df[mask]['actual']
    pred_lr = X_test_df[mask]['predicted_lr']
    pred_dt = X_test_df[mask]['predicted_dt']

    print(f"\n {gender_name} (n={mask.sum()}):")
    print(f"  {'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print(f"  {'-'*60}")
    print(f"  {'Logistic Regression':<25} "
          f"{round(accuracy_score(actual, pred_lr), 2):<12}"
          f"{round(precision_score(actual, pred_lr, zero_division=0), 2):<12}"
          f"{round(recall_score(actual, pred_lr, zero_division=0), 2):<12}"
          f"{round(f1_score(actual, pred_lr, zero_division=0), 2):<12}")
    print(f"  {'Decision Tree':<25} "
          f"{round(accuracy_score(actual, pred_dt), 2):<12}"
          f"{round(precision_score(actual, pred_dt, zero_division=0), 2):<12}"
          f"{round(recall_score(actual, pred_dt, zero_division=0), 2):<12}"
          f"{round(f1_score(actual, pred_dt, zero_division=0), 2):<12}")


# In[70]:


print("Fairness Analysis by Race:")
print("=" * 60)

# Get unique race codes
race_codes = X_test_df['race'].unique()
race_names = df_model['race'].unique()

for race_code in sorted(race_codes):
    mask = X_test_df['race'] == race_code
    actual = X_test_df[mask]['actual']
    pred_lr = X_test_df[mask]['predicted_lr']
    pred_dt = X_test_df[mask]['predicted_dt']

    print(f"\n Race Code {race_code} (n={mask.sum()}):")
    print(f"  {'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print(f"  {'-'*60}")
    print(f"  {'Logistic Regression':<25} "
          f"{round(accuracy_score(actual, pred_lr), 2):<12}"
          f"{round(precision_score(actual, pred_lr, zero_division=0), 2):<12}"
          f"{round(recall_score(actual, pred_lr, zero_division=0), 2):<12}"
          f"{round(f1_score(actual, pred_lr, zero_division=0), 2):<12}")
    print(f"  {'Decision Tree':<25} "
          f"{round(accuracy_score(actual, pred_dt), 2):<12}"
          f"{round(precision_score(actual, pred_dt, zero_division=0), 2):<12}"
          f"{round(recall_score(actual, pred_dt, zero_division=0), 2):<12}"
          f"{round(f1_score(actual, pred_dt, zero_division=0), 2):<12}")


# In[74]:


print("Fairness Analysis by Age Group:")
print("=" * 60)

# Add age group back to test set
X_test_df['age_group'] = pd.cut(X_test_df['age'],
                                  bins=[0, 25, 35, 45, 55, 65, 100],
                                  labels=['<25', '25-35', '35-45',
                                          '45-55', '55-65', '65+'])

for age_group in ['<25', '25-35', '35-45', '45-55', '55-65', '65+']:
    mask = X_test_df['age_group'] == age_group
    actual = X_test_df[mask]['actual']
    pred_lr = X_test_df[mask]['predicted_lr']
    pred_dt = X_test_df[mask]['predicted_dt']

    if len(actual) == 0:
        continue

    print(f"\n Age Group {age_group} (n={mask.sum()}):")
    print(f"  {'Model':<25} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print(f"  {'-'*60}")
    print(f"  {'Logistic Regression':<25} "
          f"{round(accuracy_score(actual, pred_lr), 2):<12}"
          f"{round(precision_score(actual, pred_lr, zero_division=0), 2):<12}"
          f"{round(recall_score(actual, pred_lr, zero_division=0), 2):<12}"
          f"{round(f1_score(actual, pred_lr, zero_division=0), 2):<12}")
    print(f"  {'Decision Tree':<25} "
          f"{round(accuracy_score(actual, pred_dt), 2):<12}"
          f"{round(precision_score(actual, pred_dt, zero_division=0), 2):<12}"
          f"{round(recall_score(actual, pred_dt, zero_division=0), 2):<12}"
          f"{round(f1_score(actual, pred_dt, zero_division=0), 2):<12}")


# In[76]:


gender_metrics = []

for gender_code, gender_name in [(0, 'Female'), (1, 'Male')]:
    mask = X_test_df['sex'] == gender_code
    actual = X_test_df[mask]['actual']
    pred_lr = X_test_df[mask]['predicted_lr']
    pred_dt = X_test_df[mask]['predicted_dt']

    gender_metrics.append({
        'Gender': gender_name,
        'LR Accuracy': round(accuracy_score(actual, pred_lr), 2),
        'LR Precision': round(precision_score(actual, pred_lr, zero_division=0), 2),
        'LR Recall': round(recall_score(actual, pred_lr, zero_division=0), 2),
        'LR F1': round(f1_score(actual, pred_lr, zero_division=0), 2),
        'DT Accuracy': round(accuracy_score(actual, pred_dt), 2),
        'DT Precision': round(precision_score(actual, pred_dt, zero_division=0), 2),
        'DT Recall': round(recall_score(actual, pred_dt, zero_division=0), 2),
        'DT F1': round(f1_score(actual, pred_dt, zero_division=0), 2)
    })

gender_df = pd.DataFrame(gender_metrics)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# LR metrics by gender
gender_df.set_index('Gender')[['LR Accuracy', 'LR Precision', 
                                'LR Recall', 'LR F1']].plot(
    kind='bar', ax=axes[0], color=['steelblue', 'green', 'orange', 'red'],
    edgecolor='black', alpha=0.8)
axes[0].set_title('Logistic Regression - Metrics by Gender')
axes[0].set_ylabel('Score')
axes[0].set_ylim(0, 1.1)
axes[0].tick_params(axis='x', rotation=0)
axes[0].grid(axis='y', linestyle='--', alpha=0.5)

# DT metrics by gender
gender_df.set_index('Gender')[['DT Accuracy', 'DT Precision',
                                'DT Recall', 'DT F1']].plot(
    kind='bar', ax=axes[1], color=['steelblue', 'green', 'orange', 'red'],
    edgecolor='black', alpha=0.8)
axes[1].set_title('Decision Tree - Metrics by Gender')
axes[1].set_ylabel('Score')
axes[1].set_ylim(0, 1.1)
axes[1].tick_params(axis='x', rotation=0)
axes[1].grid(axis='y', linestyle='--', alpha=0.5)

plt.suptitle('Fairness Analysis by Gender', fontsize=14)
plt.tight_layout()
plt.show()


# In[78]:


race_metrics = []

for race_code in sorted(X_test_df['race'].unique()):
    mask = X_test_df['race'] == race_code
    actual = X_test_df[mask]['actual']
    pred_lr = X_test_df[mask]['predicted_lr']

    race_metrics.append({
        'Race': f'Race {race_code}',
        'Accuracy': round(accuracy_score(actual, pred_lr), 2),
        'Precision': round(precision_score(actual, pred_lr, zero_division=0), 2),
        'Recall': round(recall_score(actual, pred_lr, zero_division=0), 2),
        'F1': round(f1_score(actual, pred_lr, zero_division=0), 2)
    })

race_df = pd.DataFrame(race_metrics)

race_df.set_index('Race').plot(kind='bar', figsize=(12, 5),
                                color=['steelblue', 'green', 'orange', 'red'],
                                edgecolor='black', alpha=0.8)
plt.title('Logistic Regression - Fairness by Race')
plt.ylabel('Score')
plt.ylim(0, 1.1)
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()


# In[80]:


from sklearn.utils.class_weight import compute_sample_weight
import numpy as np

# Compute weights based on sex and race combined
# This gives more weight to underrepresented groups
sample_weights = compute_sample_weight(
    class_weight='balanced',
    y=y_train)

print("Sample Weights computed!")
print(f"Min weight : {sample_weights.min():.3f}")
print(f"Max weight : {sample_weights.max():.3f}")
print(f"Mean weight: {sample_weights.mean():.3f}")


# In[82]:


# Get protected attributes from training set
X_train_df = X_train.copy()

# Assign higher weights to underrepresented groups
custom_weights = np.ones(len(X_train_df))

# Upweight females (sex=0)
custom_weights[X_train_df['sex'] == 0] *= 2.0

# Upweight minority races
minority_races = X_train_df['race'].value_counts()
minority_race_codes = minority_races[
    minority_races < minority_races.mean()].index

for race_code in minority_race_codes:
    custom_weights[X_train_df['race'] == race_code] *= 2.0

# Upweight elderly (age > 60)
custom_weights[X_train_df['age'] > 60] *= 1.5

# Normalize weights
custom_weights = custom_weights / custom_weights.mean()

print("Custom weights computed!")
print(f"Min weight : {custom_weights.min():.3f}")
print(f"Max weight : {custom_weights.max():.3f}")
print(f"Mean weight: {custom_weights.mean():.3f}")
print(f"\nWeight distribution:")
print(f"  Regular samples  : {sum(custom_weights == 1.0)}")
print(f"  Upweighted samples: {sum(custom_weights > 1.0)}")


# In[84]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

# Retrain Logistic Regression with weights
lr_fair = LogisticRegression(max_iter=10000, random_state=42)
lr_fair.fit(X_train_scaled, y_train, sample_weight=custom_weights)
y_pred_lr_fair = lr_fair.predict(X_test_scaled)

# Retrain Decision Tree with weights
dt_fair = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_fair.fit(X_train, y_train, sample_weight=custom_weights)
y_pred_dt_fair = dt_fair.predict(X_test)

print("Models retrained with fairness weights!")
print("\nLogistic Regression (Fair):")
print(f"Accuracy: {round(accuracy_score(y_test, y_pred_lr_fair), 2)}")
print(classification_report(y_test, y_pred_lr_fair,
      target_names=['<=50K', '>50K']))

print("\nDecision Tree (Fair):")
print(f"Accuracy: {round(accuracy_score(y_test, y_pred_dt_fair), 2)}")
print(classification_report(y_test, y_pred_dt_fair,
      target_names=['<=50K', '>50K']))


# In[86]:


from sklearn.metrics import precision_score, recall_score, f1_score

X_test_df['predicted_lr_fair'] = y_pred_lr_fair
X_test_df['predicted_dt_fair'] = y_pred_dt_fair

gender_fair_metrics = []

for gender_code, gender_name in [(0, 'Female'), (1, 'Male')]:
    mask = X_test_df['sex'] == gender_code
    actual = X_test_df[mask]['actual']

    # Before mitigation
    pred_lr_before = X_test_df[mask]['predicted_lr']
    # After mitigation
    pred_lr_after = X_test_df[mask]['predicted_lr_fair']

    gender_fair_metrics.append({
        'Gender': gender_name,
        'LR F1 Before': round(f1_score(actual, pred_lr_before, zero_division=0), 2),
        'LR F1 After': round(f1_score(actual, pred_lr_after, zero_division=0), 2),
        'LR Recall Before': round(recall_score(actual, pred_lr_before, zero_division=0), 2),
        'LR Recall After': round(recall_score(actual, pred_lr_after, zero_division=0), 2),
        'LR Precision Before': round(precision_score(actual, pred_lr_before, zero_division=0), 2),
        'LR Precision After': round(precision_score(actual, pred_lr_after, zero_division=0), 2),
    })

gender_fair_df = pd.DataFrame(gender_fair_metrics)
print("Gender Fairness - Before vs After Mitigation:")
print(gender_fair_df.to_string(index=False))


# In[88]:


race_fair_metrics = []

for race_code in sorted(X_test_df['race'].unique()):
    mask = X_test_df['race'] == race_code
    actual = X_test_df[mask]['actual']
    pred_lr_before = X_test_df[mask]['predicted_lr']
    pred_lr_after = X_test_df[mask]['predicted_lr_fair']

    race_fair_metrics.append({
        'Race': f'Race {race_code}',
        'F1 Before': round(f1_score(actual, pred_lr_before, zero_division=0), 2),
        'F1 After': round(f1_score(actual, pred_lr_after, zero_division=0), 2),
        'Recall Before': round(recall_score(actual, pred_lr_before, zero_division=0), 2),
        'Recall After': round(recall_score(actual, pred_lr_after, zero_division=0), 2),
    })

race_fair_df = pd.DataFrame(race_fair_metrics)
print("Race Fairness - Before vs After Mitigation:")
print(race_fair_df.to_string(index=False))


# In[90]:


fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Gender F1 Before vs After
gender_fair_df.set_index('Gender')[['LR F1 Before', 'LR F1 After']].plot(
    kind='bar', ax=axes[0,0],
    color=['steelblue', 'orange'],
    edgecolor='black', alpha=0.8)
axes[0,0].set_title('F1 Score by Gender - Before vs After')
axes[0,0].set_ylabel('F1 Score')
axes[0,0].set_ylim(0, 1.1)
axes[0,0].tick_params(axis='x', rotation=0)
axes[0,0].grid(axis='y', linestyle='--', alpha=0.5)

# Gender Recall Before vs After
gender_fair_df.set_index('Gender')[['LR Recall Before', 'LR Recall After']].plot(
    kind='bar', ax=axes[0,1],
    color=['steelblue', 'orange'],
    edgecolor='black', alpha=0.8)
axes[0,1].set_title('Recall by Gender - Before vs After')
axes[0,1].set_ylabel('Recall')
axes[0,1].set_ylim(0, 1.1)
axes[0,1].tick_params(axis='x', rotation=0)
axes[0,1].grid(axis='y', linestyle='--', alpha=0.5)

# Race F1 Before vs After
race_fair_df.set_index('Race')[['F1 Before', 'F1 After']].plot(
    kind='bar', ax=axes[1,0],
    color=['steelblue', 'orange'],
    edgecolor='black', alpha=0.8)
axes[1,0].set_title('F1 Score by Race - Before vs After')
axes[1,0].set_ylabel('F1 Score')
axes[1,0].set_ylim(0, 1.1)
axes[1,0].tick_params(axis='x', rotation=45)
axes[1,0].grid(axis='y', linestyle='--', alpha=0.5)

# Race Recall Before vs After
race_fair_df.set_index('Race')[['Recall Before', 'Recall After']].plot(
    kind='bar', ax=axes[1,1],
    color=['steelblue', 'orange'],
    edgecolor='black', alpha=0.8)
axes[1,1].set_title('Recall by Race - Before vs After')
axes[1,1].set_ylabel('Recall')
axes[1,1].set_ylim(0, 1.1)
axes[1,1].tick_params(axis='x', rotation=45)
axes[1,1].grid(axis='y', linestyle='--', alpha=0.5)

plt.suptitle('Fairness Mitigation - Before vs After Reweighting', fontsize=14)
plt.tight_layout()
plt.show()


# In[94]:


print("Overall Model Comparison:")
print("=" * 70)

comparison = pd.DataFrame({
    'Model': [
        'LR - No Fairness',
        'LR - With Fairness',
        'DT - No Fairness',
        'DT - With Fairness'
    ],
    'Accuracy': [
        round(accuracy_score(y_test, y_pred_lr), 2),
        round(accuracy_score(y_test, y_pred_lr_fair), 2),
        round(accuracy_score(y_test, y_pred_dt), 2),
        round(accuracy_score(y_test, y_pred_dt_fair), 2)
    ],
    'Precision': [
        round(precision_score(y_test, y_pred_lr, zero_division=0), 2),
        round(precision_score(y_test, y_pred_lr_fair, zero_division=0), 2),
        round(precision_score(y_test, y_pred_dt, zero_division=0), 2),
        round(precision_score(y_test, y_pred_dt_fair, zero_division=0), 2)
    ],
    'Recall': [
        round(recall_score(y_test, y_pred_lr, zero_division=0), 2),
        round(recall_score(y_test, y_pred_lr_fair, zero_division=0), 2),
        round(recall_score(y_test, y_pred_dt, zero_division=0), 2),
        round(recall_score(y_test, y_pred_dt_fair, zero_division=0), 2)
    ],
    'F1 Score': [
        round(f1_score(y_test, y_pred_lr, zero_division=0), 2),
        round(f1_score(y_test, y_pred_lr_fair, zero_division=0), 2),
        round(f1_score(y_test, y_pred_dt, zero_division=0), 2),
        round(f1_score(y_test, y_pred_dt_fair, zero_division=0), 2)
    ]
})

print(comparison.to_string(index=False))

print("\n Note:")
print("  Fairness mitigation may slightly reduce overall accuracy")
print("  but improves equity across demographic groups")
print("  This is the accuracy-fairness tradeoff!")

