# Social Network Ads - Purchase Prediction with K-Nearest Neighbors (KNN)

A beginner-friendly machine learning tutorial that predicts whether a social network user will purchase a product based on their age, gender, and estimated salary.

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
**What this does in real life:** Imagine you are getting ready to cook and you set out your cutting board, chef knife, mixing bowl, and recipe book on the counter before starting. In Python, these libraries are our pre-made digital kitchen tools for numbers, tables, and charts.  
**Why we do this:** It loads pre-built data manipulation and graphing tools so we do not have to write math or graphics code from scratch.  
**Copy this code:**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

---

### Cell 2: Load the Dataset and View First 5 Rows
**What this does in real life:** This is like downloading a store's customer ledger from the cloud and opening it on your screen to read the top 5 receipts. It shows you the columns and gives an immediate first look at the shopper records.  
**Why we do this:** We load the remote CSV file into a pandas DataFrame and display its first five rows to verify the data arrived correctly.  
**Copy this code:**
```python
url = "https://raw.githubusercontent.com/Avik-Jain/100-Days-Of-ML-Code/master/datasets/Social_Network_Ads.csv"
df = pd.read_csv(url)
df.head()
```

---

### Cell 3: Check Dataset Dimensions (Shape)
**What this does in real life:** Imagine counting how many rows of chairs and columns of desks are arranged in a classroom before students arrive. It tells you the exact grid size of your data.  
**Why we do this:** We check how many total records (rows) and attributes (columns) are present in the table.  
**Copy this code:**
```python
df.shape
```

---

### Cell 4: List All Column Names
**What this does in real life:** Think of this like reading the header labels printed across the top of each column in a paper attendance register.  
**Why we do this:** It outputs an exact Python list of all feature names so we know how each column is titled.  
**Copy this code:**
```python
df.columns.tolist()
```

---

### Cell 5: Check Duplicates and Drop User ID
**What this does in real life:** Imagine sorting through membership cards and finding that every member has a random ID number stamped on it. Because a random ID has nothing to do with whether a person buys products, we throw that number away.  
**Why we do this:** We count duplicates with the ID present and drop `User ID` because random identifier numbers provide zero predictive value.  
**Copy this code:**
```python
# Check duplicates before dropping User ID
print("Duplicates with User ID:", df.duplicated().sum())
df.drop(columns=["User ID"], inplace=True)
```

---

### Cell 6: Preview 5 Random Sample Rows
**What this does in real life:** Instead of only looking at the very first few rows, this is like pulling 5 random customer slips out of a raffle drum to get an unbiased glance at different people in the crowd.  
**Why we do this:** Sampling random rows helps spot diverse values across the dataset rather than just the top rows.  
**Copy this code:**
```python
df.sample(5, random_state=42)
```

---

### Cell 7: Re-check Dataset Shape After Dropping Column
**What this does in real life:** After tearing off the unnecessary ID column, you measure the table again to make sure exactly one column was removed and all 400 customer records remain intact.  
**Why we do this:** It verifies that dropping `User ID` reduced the column count from 5 down to 4 without losing any rows.  
**Copy this code:**
```python
df.shape
```

---

### Cell 8: Check Data Types and Non-Null Counts
**What this does in real life:** This is like checking an employee form to see which questions were answered in letters (like "Male"/"Female") and which in numbers (like age and salary), while making sure no entries were left empty.  
**Why we do this:** It inspects each column's memory usage, data type, and confirms that there are 400 non-null values in every column.  
**Copy this code:**
```python
df.info()
```

---

### Cell 9: Encode Categorical Gender into Numeric Values
**What this does in real life:** Imagine replacing the written words "Female" and "Male" on customer badges with simple numbers `0` and `1` so an electronic turnstile counter can process them automatically.  
**Why we do this:** Machine learning models only understand numbers, so `LabelEncoder` translates text categories into 0s and 1s.  
**Copy this code:**
```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["Gender"] = le.fit_transform(df["Gender"])
print(dict(zip(le.classes_, le.transform(le.classes_))))
```

---

### Cell 10: Verify Gender Column Data Type
**What this does in real life:** This is like double-checking the turnstile screen to confirm it now registers badges as numeric digits rather than text letters.  
**Why we do this:** We verify that `Gender` has been successfully converted into an integer type.  
**Copy this code:**
```python
df["Gender"].dtype
```

---

### Cell 11: Sample 5 Rows to Verify Encoded Gender
**What this does in real life:** You pull 5 random customer cards again to inspect them with your own eyes and confirm that the gender column now shows `0`s and `1`s instead of words.  
**Why we do this:** It visually confirms that label encoding was applied directly and correctly to the active dataframe.  
**Copy this code:**
```python
df.sample(5, random_state=42)
```

---

### Cell 12: Check for Missing (Null) Values
**What this does in real life:** Think of this like checking every single box on 400 job application forms to verify that not a single applicant left a question blank.  
**Why we do this:** Algorithms can fail on empty cells, so we verify that the sum of null values across all columns is zero.  
**Copy this code:**
```python
df.isnull().sum()
```

---

### Cell 13: Summary Statistics (Transposed)
**What this does in real life:** This is like calculating the average age, the lowest salary, the highest salary, and the middle values for all your shoppers to understand what an ordinary customer looks like.  
**Why we do this:** `describe().T` gives us the count, mean, standard deviation, minimum, and quartiles for every numeric feature in a readable orientation.  
**Copy this code:**
```python
df.describe().T
```

---

### Cell 14: Check Duplicate Rows Without User ID
**What this does in real life:** Now that random user IDs are gone, you check if any two people have identical gender, age, salary, and purchase history.  
**Why we do this:** It determines if there are duplicate feature records in our cleaned dataset.  
**Copy this code:**
```python
df.duplicated().sum()
```

---

### Cell 15: Display Target Class Balance (Purchased vs Not Purchased)
**What this does in real life:** Imagine tallying how many total customers in your store bought an item versus how many walked out empty-handed.  
**Why we do this:** We check the class distribution of our target variable `Purchased` to see if our dataset is balanced or skewed.  
**Copy this code:**
```python
display(df["Purchased"].value_counts().sort_index().rename(
    index={0: "Did not purchase (0)", 1: "Purchased (1)"}
))
```

---

### Cell 16: Bar Plot and Percentage Breakdown of Purchases
**What this does in real life:** This is like presenting a clear bar chart to your marketing manager showing that roughly 64% of visitors didn't buy, while about 36% bought the product.  
**Why we do this:** Visualizing the target frequency gives immediate visual and percentage confirmation of the class balance.  
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
```

---

### Cell 17: Histogram of Customer Age Distribution
**What this does in real life:** Imagine lining up all your shoppers by age and seeing that most customers are between 30 and 45 years old, with fewer very young or elderly shoppers.  
**Why we do this:** A histogram with a smooth density curve shows the spread, peaks, and symmetry of customer ages.  
**Copy this code:**
```python
# Age distribution
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x="Age", bins=20, kde=True)
plt.title("Distribution of Customer Age")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
```

---

### Cell 18: Histogram of Estimated Salary Distribution
**What this does in real life:** This is like grouping your shoppers into income brackets to see how many make \$20k, \$50k, \$80k, or over \$120k per year.  
**Why we do this:** It reveals the distribution of estimated annual salaries across the customer base.  
**Copy this code:**
```python
# Salary Distribution
plt.figure(figsize=(8, 4))
sns.histplot(data=df, x="EstimatedSalary", bins=25, kde=True)
plt.title("Distribution of Estimated Salary")
plt.xlabel("Estimated salary")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
```

---

### Cell 19: Scatter Plot of Age vs Salary Colored by Purchase
**What this does in real life:** Imagine placing every shopper as a colored pin on a 2D map (horizontal is age, vertical is salary). You immediately see that older shoppers and high earners cluster together as buyers.  
**Why we do this:** It uncovers the clear visual boundary separating buyers from non-buyers based on both age and salary together.  
**Copy this code:**
```python
plt.figure(figsize=(9, 6))
sns.scatterplot(
    data=df, x="Age", y="EstimatedSalary",
    hue="Purchased", style="Purchased", alpha=0.8
)
plt.title("Age vs Estimated Salary (colored by purchase outcome)")
plt.tight_layout()
plt.show()
```

---

### Cell 20: Correlation Heatmap
**What this does in real life:** This is like creating a score grid that reveals which factors move together—showing that age has the strongest positive correlation with buying the item.  
**Why we do this:** Heatmaps show the Pearson correlation coefficient between every pair of numeric features.  
**Copy this code:**
```python
plt.figure(figsize=(7, 5))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm",
            center=0, fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()
```

---

### Cell 21: Separate Features (X) and Target (y)
**What this does in real life:** Imagine splitting your flashcards into two piles: one pile with the customer's background details (age, salary, gender) and one pile with the answer (whether they bought).  
**Why we do this:** Machine learning models require input features (`X`) separated from the target label (`y`).  
**Copy this code:**
```python
X = df[["Gender", "Age", "EstimatedSalary"]]
y = df["Purchased"]
```

---

### Cell 22: Inspect First 5 Rows of Features (X)
**What this does in real life:** You look at the top 5 question cards to confirm they only contain gender, age, and salary.  
**Why we do this:** It verifies that `X` has the expected predictor columns without the target column.  
**Copy this code:**
```python
X.head()
```

---

### Cell 23: Inspect First 5 Values of Target (y)
**What this does in real life:** You look at the top 5 answer cards to confirm they only contain 0s and 1s representing purchases.  
**Why we do this:** It verifies that `y` is a clean series containing only the target outcomes.  
**Copy this code:**
```python
y.head()
```

---

### Cell 24: Import train_test_split
**What this does in real life:** You bring out a certified card-shuffling tool designed to divide a deck of flashcards fairly into study cards and test cards.  
**Why we do this:** It imports scikit-learn's standardized data-splitting function.  
**Copy this code:**
```python
from sklearn.model_selection import train_test_split
```

---

### Cell 25: Split Data into Train and Test Sets (80/20 Stratified)
**What this does in real life:** You put 80% of the cards into a training stack for the student to learn from, and hide 20% away in a locked drawer for exam day, ensuring both stacks have the exact same percentage of buyers.  
**Why we do this:** Stratified splitting ensures that the training and testing sets maintain the same ratio of buyers and non-buyers as the original data.  
**Copy this code:**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

---

### Cell 26: Print Row Counts of Train and Test Sets
**What this does in real life:** You count both stacks of cards to confirm you have exactly 320 practice questions and 80 exam questions.  
**Why we do this:** It confirms the 80/20 split resulted in 320 training rows and 80 test rows.  
**Copy this code:**
```python
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
```

---

### Cell 27: Import StandardScaler
**What this does in real life:** You bring out a standard converter tool so that numbers measured in tens (age 20-60) and numbers measured in tens of thousands (salary \$20,000-\$150,000) can speak the same language.  
**Why we do this:** Distance-based algorithms like KNN are biased toward large numbers unless features are scaled to mean=0 and variance=1.  
**Copy this code:**
```python
from sklearn.preprocessing import StandardScaler
```

---

### Cell 28: Fit Scaler on Train Set and Transform Train and Test
**What this does in real life:** Imagine calculating the average salary and age on your study material only, and using those exact same reference marks to scale both your study sheets and future test papers fairly.  
**Why we do this:** Fitting only on `X_train` prevents data leakage from the test set into the model.  
**Copy this code:**
```python
scaler = StandardScaler()

# Make copies, then scale only Age and EstimatedSalary (fit on train only)
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

numeric_columns = ["Age", "EstimatedSalary"]

X_train_scaled[numeric_columns] = scaler.fit_transform(X_train[numeric_columns])
X_test_scaled[numeric_columns] = scaler.transform(X_test[numeric_columns])
```

---

### Cell 29: Import KNN Classifier and Evaluation Metrics
**What this does in real life:** You open the algorithmic playbook to pull out the K-Nearest Neighbors classifier and standard score-calculating formulas.  
**Why we do this:** It imports the model class and scoring functions (accuracy, report, confusion matrix) from scikit-learn.  
**Copy this code:**
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
```

---

### Cell 30: Initialize and Train KNN Model (K=5)
**What this does in real life:** Imagine telling your AI assistant: "Memorize these 320 customer examples. Whenever you see a person, find the 5 past shoppers who are closest to them in age and salary and take a majority vote."  
**Why we do this:** It instantiates a KNN classifier with $K=5$ and fits it on the scaled training features and labels.  
**Copy this code:**
```python
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
```

---

### Cell 31: Make Predictions on Unseen Test Set
**What this does in real life:** You hand your trained assistant the 80 test questions and ask it to guess whether each person bought the product without looking at the answer key.  
**Why we do this:** It uses the fitted model to generate predicted class labels for the 80 test set customers.  
**Copy this code:**
```python
y_pred = knn.predict(X_test_scaled)
y_pred
```

---

### Cell 32: Print Accuracy Score and Classification Report
**What this does in real life:** You grade the 80 exam answers against the answer key, calculating the exact percentage score, precision, and recall.  
**Why we do this:** It evaluates how often the model is correct overall and measures performance on both classes individually.  
**Copy this code:**
```python
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy (K=5): {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

---

### Cell 33: Plot Confusion Matrix Heatmap
**What this does in real life:** This is like a 4-box scoreboard showing exactly how many non-buyers were correctly identified, how many buyers were correctly identified, and where mistakes were made.  
**Why we do this:** Visualizing the confusion matrix displays true positives, true negatives, false positives, and false negatives in a clear visual format.  
**Copy this code:**
```python
plt.figure(figsize=(6, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix (K=5)")
plt.tight_layout()
plt.show()
```

---

### Cell 34: Optimize K via 5-Fold Cross-Validation and Evaluate Best Model
**What this does in real life:** Imagine running miniature practice tournaments testing 1 neighbor, 3 neighbors, 5 neighbors, all the way up to 19 neighbors to scientifically discover which number gives the highest test score without guessing.  
**Why we do this:** Cross-validation tests odd values of $K$ inside a leak-free pipeline, trains the final model using the winning $K$, and measures test accuracy.  
**Copy this code:**
```python
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
```

---

### Cell 35: Predict Purchase for a Brand New Customer
**What this does in real life:** A brand new 30-year-old male walks into your digital store earning \$60,000. You scale his numbers, feed them into your best model, and it immediately outputs whether he is likely to buy.  
**Why we do this:** It shows how to use the final trained model in production to make a decision on brand new customer data.  
**Copy this code:**
```python
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
