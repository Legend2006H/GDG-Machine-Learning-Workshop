# Titanic Survival Prediction with Logistic Regression

A beginner-friendly machine learning tutorial that predicts passenger survival aboard the Titanic based on class, age, gender, and ticket fare.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## How to Use This README

- **Step 1:** Open [Google Colab](https://colab.research.google.com/).
- **Step 2:** Copy code from **Step 1**, paste it into a cell, and press `Shift + Enter` to run.
- **Step 3:** Repeat for Step 2, 3, 4... in order. Do not skip any steps.
- **Step 4:** The dataset in this tutorial is loaded directly from a public URL in Step 2, so no manual file upload is required.

---

## Step-by-Step Tutorial

### Step 1: Import Libraries
**What this does in real life:** Imagine laying out your toolbelt and measuring tools before starting a carpentry project. In Python, these libraries give us pre-built functions to load tables, perform math, and create visual graphs.  
**Why we do this:** It loads the essential libraries into our Python session so we don't have to build data tools from scratch.  
**Copy this code:**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

---

### Step 2: Load the Dataset
**What this does in real life:** This is like opening an old passenger manifest ledger from a ship to view the first five travelers, checking where they were seated, how much they paid, and whether they made it safely.  
**Why we do this:** We load the raw passenger data from an online CSV link into a pandas DataFrame so Python can process it.  
**Copy this code:**
```python
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
df.head()
```

---

### Step 3: Understand the Data
**What this does in real life:** Think of this like taking an initial inventory of your ledger. You count how many total passengers are recorded, inspect which columns exist, search for duplicate tickets, and draw a heatmap to spot missing entries like blank ages or cabin numbers.  
**Why we do this:** It lets us discover the shape, data types, duplicate counts, and gaps in our dataset before making changes.  
**Copy this code:**
```python
df.shape
df.isnull().sum()

# Visualizing missing values
plt.figure(figsize=(12, 6))

sns.heatmap(df.isnull(), cbar=False, cmap='viridis')

plt.title("Missing Values Heatmap")
plt.show()

df.info()
df.columns.tolist()
df.duplicated().sum()
df[df.duplicated()]

# Separating numerical and categorical columns
# Numerical Columns
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
numerical_cols

# Categorical Columns
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
categorical_cols
```

---

### Step 4: Data Cleaning
**What this does in real life:** Imagine finding blank age boxes on passenger tickets and filling them in with the average age of all passengers on board so no passenger is left with missing information. We then round those ages to clean whole numbers.  
**Why we do this:** Machine learning algorithms cannot handle missing values, so we impute missing ages with the column mean.  
**Copy this code:**
```python
# Compute mean age before imputation, then fill missing values
mean_age = df["Age"].mean()
df["Age"] = df["Age"].fillna(mean_age)

df["Age"].mean()

df["Age"] = df["Age"].round().astype("int64")
df.info()
df.isnull().sum()
```

---

### Step 5: Exploratory Data Analysis - EDA
**What this does in real life:** This is like pinning bar charts and graphs to a wall to spot real historical trends—such as seeing that 1st class passengers and women had significantly higher survival rates. We also select only the most informative columns, turn gender text into numbers (`0` for female, `1` for male), and plot feature correlations.  
**Why we do this:** Visualizing the data reveals key patterns affecting survival and ensures all selected features are encoded properly for training.  
**Copy this code:**
```python
# Survival percentage by passenger class
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='Pclass', y='Survived')

plt.title("Survival Rate by Passenger Class")
plt.ylabel("Survival Rate")
plt.show()

# Survival percentage by gender
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='Sex', y='Survived', order=['female', 'male'])

plt.title("Survival Rate by Gender")
plt.xticks([0, 1], ['Female', 'Male'])
plt.ylabel("Survival Rate")
plt.show()

# Fare vs Survival
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Survived', y='Fare')

plt.title("Fare Distribution by Survival")
plt.show()

# Selecting only required columns
df = df[['Pclass', 'Age', 'Sex', 'Fare', 'Survived']].copy()
df.head()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])
print(dict(zip(le.classes_, le.transform(le.classes_))))

df.head()

plt.figure(figsize=(12, 8))

for i, col in enumerate(df.columns, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f"Distribution of {col}")

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')

plt.title("Feature Correlation Heatmap")
plt.show()
```

---

### Step 6: Split Data into Features and Target
**What this does in real life:** This is like splitting each passenger record into two distinct index cards: one with their travel details (ticket class, age, gender, fare) and one with the historical outcome (whether they survived).  
**Why we do this:** The machine learning model requires separate matrices for predictor attributes (`X`) and the outcome label (`y`).  
**Copy this code:**
```python
X = df[["Pclass", "Age", "Sex", "Fare"]]
y = df["Survived"]

X.head()
X.shape
y.head()
y.shape
```

---

### Step 7: Train-Test Split
**What this does in real life:** Imagine studying 80% of historical cases to learn how survival was determined, while locking 20% in an envelope for a blind final exam. We also standardize the fare values so large dollar amounts don't distort the mathematical calculations.  
**Why we do this:** Splitting avoids overfitting on seen data, and standardizing fare puts it on a balanced scale with age and class.  
**Copy this code:**
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train = X_train.copy()
X_test = X_test.copy()
X_train_raw = X_train.copy()  # Kept unscaled for leak-free cross-validation

X_train.shape
y_train.shape
X_test.shape

fare_original = X_train["Fare"].copy()
fare_original.head()

from sklearn.preprocessing import StandardScaler

# Fit scaler only on training data to prevent data leakage
scaler = StandardScaler()
X_train["Fare"] = scaler.fit_transform(X_train[["Fare"]])
X_test["Fare"] = scaler.transform(X_test[["Fare"]])

X_train.head()

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Before standardization
sns.histplot(fare_original, kde=True, ax=ax[0])
ax[0].set_title("Fare Before Standardization (Train)")

# After standardization
sns.histplot(X_train['Fare'], kde=True, ax=ax[1])
ax[1].set_title("Fare After Standardization (Train)")

plt.tight_layout()
plt.show()
```

---

### Step 8: Train the Model
**What this does in real life:** Imagine teaching an insurance analyst how passenger traits influenced survival chances by showing them hundreds of real passenger cases. Logistic Regression finds the exact mathematical weights connecting passenger features to their survival probability.  
**Why we do this:** We train the Logistic Regression algorithm so it learns how class, age, sex, and fare influenced passenger survival.  
**Copy this code:**
```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)
```

---

### Step 9: Make Predictions
**What this does in real life:** This is like handing the analyst the 20% unseen passenger records and asking them to predict who survived without revealing the answers. The model predicts a `1` (survived) or `0` (did not survive) for each test passenger.  
**Why we do this:** Generating predictions on test cases allows us to test the model's accuracy on unfamiliar passengers.  
**Copy this code:**
```python
y_pred = lr.predict(X_test)
y_pred
```

---

### Step 10: Evaluate the Model
**What this does in real life:** This is like grading the exam paper against the answer key. We calculate accuracy, precision, and recall, draw a confusion matrix and an ROC curve, run 5-fold cross-validation to ensure reliable scoring, and finally predict survival for 3 hypothetical new passengers.  
**Why we do this:** It provides a comprehensive performance assessment and verifies the model's predictions on new, real-world examples.  
**Copy this code:**
```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

from sklearn.metrics import roc_curve, roc_auc_score

y_prob = lr.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

plt.figure(figsize=(8, 5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_test, y_prob):.3f}")
plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Use a Pipeline so StandardScaler is fit inside each CV fold without data leakage
cv_pipeline = Pipeline([
    ("scaler", ColumnTransformer([("fare", StandardScaler(), ["Fare"])], remainder="passthrough")),
    ("lr", LogisticRegression(random_state=42))
])

cv_scores = cross_val_score(cv_pipeline, X_train_raw, y_train, cv=5, scoring='accuracy')

print("CV Scores:", cv_scores)
print("Mean CV Accuracy:", cv_scores.mean())
print("Standard Deviation:", cv_scores.std())

new_passengers = pd.DataFrame({
    'Pclass': [1, 3, 2],
    'Age': [25, 35, 10],
    'Sex': ['female', 'male', 'female'],
    'Fare': [100, 7.25, 30]
})

# Encode gender
new_passengers['Sex'] = new_passengers['Sex'].map({'female': 0, 'male': 1})

# Apply previously fitted scaler
new_passengers['Fare'] = scaler.transform(new_passengers[['Fare']])

# Predict
predictions = lr.predict(new_passengers)
print(predictions)
```

---

### Step 11: Conclusion - What We Learned
**What this does in real life:** Looking back at the historical rescue data, we confirmed that women and higher-class passengers with premium fares had the highest likelihood of survival. Logistic regression proved to be a fast, interpretable model, and cross-validation verified consistent predictive performance across different groups of passengers.  
**Why we do this:** It consolidates the machine learning takeaways from this classification project for future work.  
*(Note: This step summarizes the project findings; no additional code execution is required)*
