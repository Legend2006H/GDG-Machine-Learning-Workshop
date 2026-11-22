# Social Network Ads - Purchase Prediction with K-Nearest Neighbors (KNN)

A beginner-friendly machine learning tutorial that predicts whether a social network user will purchase a product based on their age, gender, and estimated salary.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

---

## How to Use This README

- **Step 1:** Open [Google Colab](https://colab.research.google.com/).
- **Step 2:** Copy the code from **Step 1**, paste it into a code cell, and press `Shift + Enter` to run.
- **Step 3:** Repeat for Step 2, 3, 4... in order. Do not skip any steps.
- **Step 4:** The dataset in this tutorial is loaded directly from a public URL in Step 2, so no manual file upload is required.

---

## Step-by-Step Tutorial

### Step 1: Import Libraries
**What this does in real life:** Imagine you are about to cook a meal and you set out all your cutting boards, pans, and measuring spoons on the kitchen counter first. In Python, these libraries are our pre-made digital toolsets for organizing tables, crunching numbers, and drawing charts.  
**Why we do this:** It imports all the necessary tools so we do not have to write complex math or plotting logic from scratch.  
**Copy this code:**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

---

### Step 2: Load the Dataset
**What this does in real life:** This is like downloading a customer survey spreadsheet from the internet and opening it to look at the first five customer records. It gives us our very first peek at who our shoppers are and what information was collected about them.  
**Why we do this:** We need to load the data file from the web into Python's memory so our code can read and analyze it.  
**Copy this code:**
```python
url = "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/Social_Network_Ads.csv"
df = pd.read_csv(url)
df.head()
```

---

### Step 3: Understand the Data
**What this does in real life:** Think of this like checking the label on a newly delivered parcel. You check how many items are in the box, read the list of contents, inspect the item details, and verify that nothing was lost or broken during delivery.  
**Why we do this:** It allows us to inspect the dataset's dimensions, column names, data types, and check for missing values.  
**Copy this code:**
```python
df.shape
df.columns.tolist()
df.info()
df.isnull().sum()
```

---

### Step 4: Data Cleaning
**What this does in real life:** Imagine sorting through a stack of customer registration cards and removing random badge numbers that have no bearing on shopping habits. We also translate words like "Male" and "Female" into numbers (`1` and `0`) so our digital calculator can understand them.  
**Why we do this:** Machine learning models require clean numeric values and perform better when irrelevant identifier columns are removed.  
**Copy this code:**
```python
# Check duplicates before dropping User ID
print("Duplicates with User ID:", df.duplicated().sum())
df.drop(columns=["User ID"], inplace=True)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])
print(dict(zip(le.classes_, le.transform(le.classes_))))

df["Gender"].dtype
df.sample(5, random_state=42)
df.describe().T
df.duplicated().sum()
display(df["Purchased"].value_counts().sort_index().rename(
    index={0: "Did not purchase (0)", 1: "Purchased (1)"}
))
```

---

### Step 5: Exploratory Data Analysis - EDA
**What this does in real life:** This is like drawing colorful charts and visual maps of your customer base on a whiteboard. Looking at pictures quickly reveals that older customers with higher incomes are significantly more likely to purchase the advertised product.  
**Why we do this:** Visualizing the data uncovers patterns, distributions, and correlations between customer attributes and buying decisions.  
**Copy this code:**
```python
plt.figure(figsize=(7, 4))
ax = sns.countplot(data=df, x="Purchased", hue="Purchased")
if ax.get_legend() is not None:
    ax.get_legend().remove()
ax.set_title("Customer Purchase Distribution")
ax.set_xlabel("Purchased (0 = No, 1 = Yes)")
ax.set_ylabel("Number of customers")
plt.tight_layout()
plt.show()

print((df["Purchased"].value_counts(normalize=True).sort_index() * 100).round(1))

# Age distribution
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x="Age", bins=20, kde=True)
plt.title("Distribution of Customer Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Salary Distribution
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x="EstimatedSalary", bins=25, kde=True)
plt.title("Distribution of Estimated Salary")
plt.xlabel("Estimated salary")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df, x="Age", y="EstimatedSalary",
    hue="Purchased", style="Purchased", alpha=0.8
)
plt.title("Age vs Estimated Salary (colored by purchase outcome)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm",
            center=0, fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
```

---

### Step 6: Split Data into Features and Target
**What this does in real life:** This is like separating a student study sheet into two stacks: the clues/questions (customer age, salary, and gender) and the final answer key (whether they bought the item).  
**Why we do this:** The machine learning algorithm needs input features (`X`) separated from the target output (`y`) it tries to predict.  
**Copy this code:**
```python
X = df[["Gender", "Age", "EstimatedSalary"]]
y = df["Purchased"]

X.head()
y.head()
```

---

### Step 7: Train-Test Split
**What this does in real life:** Imagine studying 80 practice problems to prepare for an exam, while locking away 20 surprise questions to test yourself later without peeking. We also convert age (e.g., 35) and salary (e.g., $75,000) onto the same fair scale so the huge salary numbers don't unfairly drown out age.  
**Why we do this:** Splitting avoids testing on familiar data, and scaling prevents features with large numbers from dominating distance calculations.  
**Copy this code:**
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Make copies, then scale only Age and EstimatedSalary (fit on train only)
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

numeric_columns = ["Age", "EstimatedSalary"]

X_train_scaled[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
X_test_scaled[numeric_columns] = scaler.transform(X_test[numeric_columns])
```

---

### Step 8: Train the Model
**What this does in real life:** Imagine asking your 5 nearest friends for their shopping recommendations before deciding whether to make a purchase. The K-Nearest Neighbors (KNN) algorithm stores all training customer examples so it can look at the 5 most similar past customers.  
**Why we do this:** We fit the classifier on the training data so it learns the relationship between customer profiles and purchases.  
**Copy this code:**
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
```

---

### Step 9: Make Predictions
**What this does in real life:** This is like handing your trained student the 20 surprise test questions and having them write down their answers without seeing the answer key. The model predicts whether each customer will buy based on their nearest neighbors.  
**Why we do this:** We generate predictions on unseen test customers to evaluate how well the model generalizes to new data.  
**Copy this code:**
```python
y_pred = knn.predict(X_test_scaled)
y_pred
```

---

### Step 10: Evaluate the Model
**What this does in real life:** This is like grading the exam with a red pen, calculating the overall percentage score, and checking where mistakes happened. Then we test different numbers of neighbors using cross-validation to find the optimal setting, and finally test our best model on a real-life 30-year-old making $60,000.  
**Why we do this:** It shows how accurate our model is in practice and verifies its predictions on a sample customer profile.  
**Copy this code:**
```python
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy (K=5): {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (K=5)")
plt.tight_layout()
plt.show()

# Find optimal K using cross-validation on the training set (avoiding test-set leakage)
from sklearn.model_selection import cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

k_range = range(1, 21, 2)  # Odd values avoid distance ties in binary classification
cv_means = []

for k in k_range:
    pipe = Pipeline([
        ("scaler", ColumnTransformer([("num", StandardScaler(), numeric_columns)], remainder="passthrough")),
        ("knn", KNeighborsClassifier(n_neighbors=k))
    ])
    scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy")
    cv_means.append(scores.mean())

best_k = list(k_range)[int(np.argmax(cv_means))]
print(f"Best K (5-fold CV): {best_k} | Mean CV Accuracy: {max(cv_means):.2%}")

model = KNeighborsClassifier(n_neighbors=best_k)
model.fit(X_train_scaled, y_train)

predictions = model.predict(X_test_scaled)
score = accuracy_score(y_test, predictions)

print(f"Test Accuracy (K = {best_k}): {score:.2%}")

new_person = pd.DataFrame({
    "Age": [30],
    "EstimatedSalary": [60000],
    "Gender": [1]  # 1 = Male, 0 = Female
})

# Keep the same column order used during training
new_person = new_person[X.columns]

new_person_scaled = new_person.copy()
new_person_scaled[numeric_columns] = scaler.transform(new_person[numeric_columns])

prediction = model.predict(new_person_scaled)[0]

if prediction == 1:
    print("Prediction: This person may purchase the product.")
else:
    print("Prediction: This person may not purchase the product.")
```

---

### Step 11: Conclusion - What We Learned
**What this does in real life:** This is like reviewing your score report after an exam to see what study habits paid off and what to remember for next time. We saw that older customers with higher salaries are the primary buyers, feature scaling is essential for distance models, and tuning the number of neighbors produces high predictive accuracy.  
**Why we do this:** It consolidates key takeaways from the machine learning workflow so students can apply them to future projects.  
*(Note: This step summarizes the project findings; no additional code execution is required)*
