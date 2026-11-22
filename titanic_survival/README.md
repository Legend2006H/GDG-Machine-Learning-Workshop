# Titanic Survival Prediction with Logistic Regression

A beginner-friendly machine learning tutorial that predicts passenger survival aboard the Titanic based on passenger class, age, gender, and ticket fare.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## How to Use This README

- **Step 1:** Open [Google Colab](https://colab.research.google.com/).
- **Step 2:** Copy code from **Cell 1**, paste in a cell, press `Shift + Enter` to run.
- **Step 3:** Repeat for Cell 2, 3, 4... in exact order without skipping or combining cells.
- **Step 4:** The dataset is loaded directly via a public URL in Cell 2, so no manual file upload is required.

---

## Step-by-Step Cell Flow

### Cell 1: Import Essential Libraries
**What this does in real life:** Imagine laying out your toolbelt and measuring tools before starting a carpentry project. In Python, these libraries give us pre-built tools to load tables, perform math, and create visual graphs.  
**Why we do this:** It loads the essential libraries into our Python session so we don't have to build data tools from scratch.  
**Copy this code:**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

---

### Cell 2: Load the Dataset and View First 5 Rows
**What this does in real life:** This is like opening an old passenger manifest ledger from a ship to view the first five travelers, checking where they were seated, how much they paid, and whether they made it safely.  
**Why we do this:** We load the raw passenger data from an online CSV link into a pandas DataFrame and display the top 5 rows.  
**Copy this code:**
```python
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
df.head()
```

---

### Cell 3: Check Dataset Dimensions (Shape)
**What this does in real life:** Imagine counting how many lines of passenger names are written in the ledger and how many details were collected about each traveler.  
**Why we do this:** It displays the total number of rows (891 passengers) and columns (12 attributes).  
**Copy this code:**
```python
df.shape
```

---

### Cell 4: Check Missing (Null) Values Count
**What this does in real life:** Think of this like checking an inspection sheet to count how many passenger tickets had blank spaces for Age, Cabin number, or Port of Embarkation.  
**Why we do this:** It tallies the exact number of missing entries in each column.  
**Copy this code:**
```python
df.isnull().sum()
```

---

### Cell 5: Visualize Missing Values Heatmap
**What this does in real life:** This is like holding the ledger up to the light so every blank box glows in bright yellow, showing you instantly that Cabin is mostly empty and Age has scattered missing values.  
**Why we do this:** A heatmap provides an immediate visual overview of where missing data is located across all records.  
**Copy this code:**
```python
# Visualizing missing values
plt.figure(figsize=(12, 6))

sns.heatmap(df.isnull(), cbar=False, cmap='viridis')

plt.title("Missing Values Heatmap")
plt.show()
```

---

### Cell 6: Inspect Column Info and Data Types
**What this does in real life:** This is like scanning through the ledger's technical index to see which fields are numbers, which are names, and how much computer memory the book takes up.  
**Why we do this:** `df.info()` confirms the data types and non-null counts of every column.  
**Copy this code:**
```python
df.info()
```

---

### Cell 7: Impute Missing Age Values with the Mean Age
**What this does in real life:** Imagine finding a passenger ticket with no age recorded, and politely writing in the average age of all other passengers so nobody is left blank.  
**Why we do this:** We fill missing values in the `Age` column with the column mean so models don't drop those passengers.  
**Copy this code:**
```python
# Compute mean age before imputation, then fill missing values
mean_age = df["Age"].mean()
df["Age"] = df["Age"].fillna(mean_age)
```

---

### Cell 8: Check Mean Age Value
**What this does in real life:** You check the calculator screen to verify what that average passenger age was (approximately 29.7 years).  
**Why we do this:** It prints the exact mathematical average of the age column.  
**Copy this code:**
```python
df["Age"].mean()
```

---

### Cell 9: Round Age and Convert to Integer
**What this does in real life:** Nobody says they are "29.699 years old" on a ticket, so you round every passenger's age to a sensible whole number.  
**Why we do this:** It rounds decimal ages to integers for cleaner, more realistic passenger ages.  
**Copy this code:**
```python
df["Age"] = df["Age"].round().astype("int64")
```

---

### Cell 10: Re-check Column Info After Age Imputation
**What this does in real life:** You inspect the ledger index again to verify that `Age` now has 891 valid entries and has been turned into a clean integer type.  
**Why we do this:** It confirms that the `Age` column has zero nulls remaining and has the `int64` data type.  
**Copy this code:**
```python
df.info()
```

---

### Cell 11: List Column Names
**What this does in real life:** You quickly read through all column headers across the page to decide which ones you need.  
**Why we do this:** It outputs a list of all column headers in the dataframe.  
**Copy this code:**
```python
df.columns.tolist()
```

---

### Cell 12: Check Missing Values Count Again
**What this does in real life:** You run a second check on your ledger to confirm that `Age` no longer has any missing entries.  
**Why we do this:** It verifies that the null count for `Age` dropped to 0.  
**Copy this code:**
```python
df.isnull().sum()
```

---

### Cell 13: Check for Duplicate Rows
**What this does in real life:** You check if any duplicate ticket stubs were accidentally entered twice into the book.  
**Why we do this:** It counts the number of completely duplicated rows in the dataset.  
**Copy this code:**
```python
df.duplicated().sum()
```

---

### Cell 14: Filter and View Duplicate Rows
**What this does in real life:** You look at the specific duplicate entries to inspect them (showing an empty result because every ticket is unique).  
**Why we do this:** It displays the duplicate records if any exist.  
**Copy this code:**
```python
df[df.duplicated()]
```

---

### Cell 15: Separate and List Numerical Columns
**What this does in real life:** You make a list of every question answered with numbers (like age, fare, number of siblings, and passenger class).  
**Why we do this:** It programmatically selects all numeric column names for analysis.  
**Copy this code:**
```python
# Separating numerical and categorical columns
# Numerical Columns
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
numerical_cols
```

---

### Cell 16: Separate and List Categorical Columns
**What this does in real life:** You make a list of every question answered with words (like passenger name, sex, ticket code, cabin, and home port).  
**Why we do this:** It programmatically selects all text/object column names.  
**Copy this code:**
```python
# Categorical Columns
categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
categorical_cols
```

---

### Cell 17: Bar Plot of Survival Rate by Passenger Class
**What this does in real life:** You draw a chart comparing 1st, 2nd, and 3rd class passengers, visually proving that 1st class travelers had a much higher rate of survival.  
**Why we do this:** It visualizes the impact of socioeconomic status (`Pclass`) on survival probability.  
**Copy this code:**
```python
# Survival percentage by passenger class
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='Pclass', y='Survived')

plt.title("Survival Rate by Passenger Class")
plt.ylabel("Survival Rate")
plt.show()
```

---

### Cell 18: Bar Plot of Survival Rate by Gender
**What this does in real life:** You create a chart comparing men and women, clearly showing the "women and children first" maritime evacuation protocol in action.  
**Why we do this:** It displays the massive difference in survival rate between female and male passengers.  
**Copy this code:**
```python
# Survival percentage by gender
plt.figure(figsize=(8, 5))
sns.barplot(data=df, x='Sex', y='Survived', order=['female', 'male'])

plt.title("Survival Rate by Gender")
plt.xticks([0, 1], ['Female', 'Male'])
plt.ylabel("Survival Rate")
plt.show()
```

---

### Cell 19: Box Plot of Fare Distribution by Survival
**What this does in real life:** You compare ticket prices paid by survivors versus non-survivors, noticing that people who paid higher fares were much more likely to survive.  
**Why we do this:** A box plot reveals the median, spread, and outliers of ticket fares for each survival outcome.  
**Copy this code:**
```python
# Fare vs Survival
plt.figure(figsize=(8, 5))
sns.boxplot(data=df, x='Survived', y='Fare')

plt.title("Fare Distribution by Survival")
plt.show()
```

---

### Cell 20: Select Only Relevant Feature Columns
**What this does in real life:** You copy only the 4 most meaningful clues (Class, Age, Sex, Fare) and the outcome (Survived) into a tidy new notebook, leaving behind names and ticket numbers.  
**Why we do this:** Eliminating irrelevant columns simplifies the model and prevents overfitting on passenger names.  
**Copy this code:**
```python
# Selecting only required columns
df = df[['Pclass', 'Age', 'Sex', 'Fare', 'Survived']].copy()
df.head()
```

---

### Cell 21: Import LabelEncoder
**What this does in real life:** You pull out a certified word-to-number dictionary tool from your digital toolkit.  
**Why we do this:** It imports scikit-learn's encoder to transform text categories into numbers.  
**Copy this code:**
```python
from sklearn.preprocessing import LabelEncoder
```

---

### Cell 22: Encode Sex (female=0, male=1)
**What this does in real life:** You replace the words "female" and "male" with digits `0` and `1` so the mathematical formula can calculate with them.  
**Why we do this:** Machine learning models require numeric inputs.  
**Copy this code:**
```python
le = LabelEncoder()
df["Sex"] = le.fit_transform(df["Sex"])
print(dict(zip(le.classes_, le.transform(le.classes_))))
```

---

### Cell 23: Inspect First 5 Rows After Encoding
**What this does in real life:** You check the first 5 records to confirm that the `Sex` column now shows `0`s and `1`s.  
**Why we do this:** It visually confirms that the label encoder transformed the dataframe in place.  
**Copy this code:**
```python
df.head()
```

---

### Cell 24: Histograms of All Cleaned Features
**What this does in real life:** You draw small distribution graphs for each of your 5 columns all on one poster board to see their shapes at a glance.  
**Why we do this:** It checks the distributions and shapes of all predictor features and the target variable.  
**Copy this code:**
```python
plt.figure(figsize=(12, 8))

for i, col in enumerate(df.columns, 1):
    plt.subplot(2, 3, i)
    sns.histplot(df[col], kde=True, bins=20)
    plt.title(f"Distribution of {col}")

plt.tight_layout()
plt.show()
```

---

### Cell 25: Correlation Heatmap
**What this does in real life:** This is like a summary scorecard that shows how strongly each trait is tied to survival—showing negative correlation with Sex (men had lower survival) and positive correlation with Fare.  
**Why we do this:** It measures linear correlation between all features and the survival outcome.  
**Copy this code:**
```python
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')

plt.title("Feature Correlation Heatmap")
plt.show()
```

---

### Cell 26: Split Features (X) and Target (y)
**What this does in real life:** You divide each passenger's record into the question card (Pclass, Age, Sex, Fare) and the answer card (Survived).  
**Why we do this:** It separates independent predictors (`X`) from the dependent target (`y`).  
**Copy this code:**
```python
X = df[["Pclass", "Age", "Sex", "Fare"]]
y = df["Survived"]
```

---

### Cell 27: Preview First 5 Rows of Features (X)
**What this does in real life:** You verify the first 5 question cards to ensure they only contain the 4 predictor columns.  
**Why we do this:** It confirms that `X` is correctly formatted.  
**Copy this code:**
```python
X.head()
```

---

### Cell 28: Check Features Shape
**What this does in real life:** You count your question cards to verify you have 891 cards with 4 questions each.  
**Why we do this:** It verifies that `X` has 891 rows and 4 columns.  
**Copy this code:**
```python
X.shape
```

---

### Cell 29: Preview First 5 Values of Target (y)
**What this does in real life:** You check the first 5 answer cards to confirm they only have 0s and 1s.  
**Why we do this:** It verifies that `y` contains only the survival labels.  
**Copy this code:**
```python
y.head()
```

---

### Cell 30: Check Target Shape
**What this does in real life:** You verify that you have exactly 891 answers matching the 891 passengers.  
**Why we do this:** It ensures `y` matches the length of `X`.  
**Copy this code:**
```python
y.shape
```

---

### Cell 31: Train-Test Split (80/20 Stratified)
**What this does in real life:** You set aside 80% of passenger records for studying and lock away 20% for the final test, making sure the percentage of survivors is identical in both piles.  
**Why we do this:** Stratified splitting ensures fair, representative training and testing subsets without data leakage.  
**Copy this code:**
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train = X_train.copy()
X_test = X_test.copy()
X_train_raw = X_train.copy()  # Kept unscaled for leak-free cross-validation
```

---

### Cell 32: Check Training Features Shape
**What this does in real life:** You count your study cards to verify you have 712 passengers for training.  
**Why we do this:** It verifies that `X_train` has 712 rows and 4 columns.  
**Copy this code:**
```python
X_train.shape
```

---

### Cell 33: Check Training Target Shape
**What this does in real life:** You verify that you have 712 answers corresponding to the 712 study cards.  
**Why we do this:** It confirms `y_train` has 712 labels.  
**Copy this code:**
```python
y_train.shape
```

---

### Cell 34: Check Testing Features Shape
**What this does in real life:** You verify that you have 179 surprise exam cards saved for test day.  
**Why we do this:** It confirms `X_test` contains 179 rows.  
**Copy this code:**
```python
X_test.shape
```

---

### Cell 35: Preserve Original Fare Values for Comparison
**What this does in real life:** You write down the original dollar amounts of the ticket fares so you can compare them before and after standardization.  
**Why we do this:** Storing an unscaled copy of `Fare` lets us plot a side-by-side comparison.  
**Copy this code:**
```python
fare_original = X_train["Fare"].copy()
fare_original.head()
```

---

### Cell 36: Standardize Fare Using StandardScaler
**What this does in real life:** Because fares can be as high as \$500 while class is just 1, 2, or 3, you rescale fares so that large dollar values don't overpower the other features.  
**Why we do this:** Fitting the scaler on `X_train` and transforming `X_test` avoids test data leakage and balances feature scales.  
**Copy this code:**
```python
from sklearn.preprocessing import StandardScaler

# Fit scaler only on training data to prevent data leakage
scaler = StandardScaler()
X_train["Fare"] = scaler.fit_transform(X_train[["Fare"]])
X_test["Fare"] = scaler.transform(X_test[["Fare"]])
```

---

### Cell 37: Inspect Scaled Training Features
**What this does in real life:** You inspect your study cards again to see that fares are now expressed as centered standard scores around 0.  
**Why we do this:** It verifies that `Fare` has been updated in `X_train`.  
**Copy this code:**
```python
X_train.head()
```

---

### Cell 38: Plot Fare Before vs After Standardization
**What this does in real life:** You place two histograms side by side to prove that standardization shifted the center of the curve to 0 while keeping the original shape intact.  
**Why we do this:** Visualizing both distributions proves that scaling normalizes the range without altering the underlying distribution.  
**Copy this code:**
```python
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

### Cell 39: Train Logistic Regression Model
**What this does in real life:** Imagine having an experienced maritime investigator analyze all 712 passenger study cards to discover the exact mathematical formula that best predicts survival.  
**Why we do this:** We fit a Logistic Regression model on `X_train` and `y_train` to learn the feature coefficients.  
**Copy this code:**
```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression(random_state=42)
lr.fit(X_train, y_train)
```

---

### Cell 40: Make Predictions on Unseen Test Set
**What this does in real life:** You hand your investigator the 179 test cases and ask them to predict who survived without peeking at the answers.  
**Why we do this:** It produces model predictions (`0` for perished, `1` for survived) for the test set.  
**Copy this code:**
```python
y_pred = lr.predict(X_test)
y_pred
```

---

### Cell 41: Import Scikit-Learn Evaluation Metrics
**What this does in real life:** You take out standard grading tools to calculate accuracy, precision, recall, and draw error tables.  
**Why we do this:** It loads classification metric functions from scikit-learn.  
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
```

---

### Cell 42: Print Accuracy, Precision, Recall, and Classification Report
**What this does in real life:** You grade the 179 predictions against the actual outcomes, printing the overall percentage score and measuring how well the model caught true survivors.  
**Why we do this:** It outputs a complete quantitative assessment of the model's predictive performance.  
**Copy this code:**
```python
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

---

### Cell 43: Plot Confusion Matrix Heatmap
**What this does in real life:** This is like a 4-square scorecard showing how many survivors were correctly found, how many non-survivors were correctly identified, and where mistakes were made.  
**Why we do this:** It visualizes true positives, false positives, true negatives, and false negatives in an intuitive graphic.  
**Copy this code:**
```python
plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

---

### Cell 44: Plot ROC Curve and Compute AUC Score
**What this does in real life:** Imagine testing how well your detector works across all possible sensitivity levels, proving it is far superior to random coin-flipping.  
**Why we do this:** The ROC curve and AUC score quantify how well the model distinguishes between survivors and non-survivors across all classification thresholds.  
**Copy this code:**
```python
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
```

---

### Cell 45: Run Leak-Free 5-Fold Cross-Validation with Pipeline
**What this does in real life:** Imagine testing the model across 5 different mini-competitions on different slices of data, ensuring the scaler is recalibrated inside each slice so there is zero cheating.  
**Why we do this:** A scikit-learn Pipeline prevents data leakage during cross-validation, confirming reliable average performance across multiple folds.  
**Copy this code:**
```python
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
```

---

### Cell 46: Predict Survival for 3 New Hypothetical Passengers
**What this does in real life:** Three new passengers step onto the ship: a wealthy 25-year-old woman in 1st class, a 35-year-old man with a cheap 3rd class ticket, and a 10-year-old girl in 2nd class. The model uses their traits to predict who survives.  
**Why we do this:** It demonstrates how to apply the trained model to new real-world data in practice.  
**Copy this code:**
```python
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
