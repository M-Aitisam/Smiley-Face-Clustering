# ⭐ Smiley Face Clustering & Eid Sales Optimization

## ⭐ Task 0: Clustering the Smiley Dataset
### 📌 Overview
You are given `smiley_dataset.csv`, which contains points that, when visualized, form a smiley face. Your task is to apply both **DBSCAN** and **K-Means** clustering on the dataset.

### 🖼️ Visualizing the Dataset
```python
import pandas as pd
import matplotlib.pyplot as plt

csv_path = "smiley_dataset.csv"
smile = pd.read_csv(csv_path)
plt.figure(figsize=(6,6))
plt.scatter(smile['X'], smile['Y'], alpha=0.7)
plt.title('Smiley Face Dataset')
plt.xlabel('X')
plt.ylabel('Y')
plt.axis('equal')
plt.show()
```

### 🔍 Clustering Parameters
- **K-Means:** `k = 4` (since we expect four clusters).
- **DBSCAN:** `eps = 0.3`, `min_samples = 5`.
- **Experiment:** Change DBSCAN parameters to observe their effects.

### 📖 Key Concepts
- **eps (epsilon)**: Defines the maximum radius around a point to consider neighbors for forming a cluster.
- **min_samples**: Specifies the minimum number of points required to form a dense region (core point).

### ❓ Questions to Answer
1. How do K-Means and DBSCAN handle the dataset differently?
2. Why might DBSCAN perform better on non-linear clusters?

---

## ⭐ Task 1: Optimize Sweet Item Placement for Eid Sales
### 📌 Overview
For the occasion of Eid, the store owner wants to **optimize the placement of sweet items**. The goal is to use the **FP-Growth algorithm** to identify frequently bought sweet items so the store owner can strategically arrange them together for increased sales.

### 🛒 Steps to Follow
1. Use the provided list of transactions representing customer purchases.
2. Apply the **FP-Growth algorithm** to find frequent itemsets.
3. Analyze the results to determine which sweet items are often bought together.
4. Explain how these insights can help the store owner optimize product placement for Eid sales.

### 🍬 List of Transactions (Eid Sweet Purchases)
```python
transactions = [
    ["gulab jamun", "barfi", "jalebi"],
    ["gulab jamun", "laddu", "halwa", "kheer"],
    ["barfi", "jalebi", "laddu"],
    ["gulab jamun", "barfi", "laddu", "jalebi"],
    ["halwa", "kheer", "soan papdi"],
    ["gulab jamun", "barfi", "jalebi", "rasmalai"],
    ["barfi", "jalebi", "soan papdi", "peda"],
    ["laddu", "kheer", "barfi"],
    ["gulab jamun", "barfi", "jalebi", "rasmalai"],
    ["halwa", "soan papdi", "kheer"]
]
```

### ❓ Questions to Answer
1. What are the **top frequent sweet itemsets** found using FP-Growth?
2. How can the store owner **arrange sweet items together** to increase Eid sales?
3. Experiment by changing the **min_support value** and observe how it affects the frequent itemsets.

---

## ⭐ Task 2: Evaluation Metrics
### 📌 Overview
A deep dive into **evaluation metrics** used in machine learning, covering **precision, accuracy, recall, and F1-score**.

### 🧮 Definitions & Formulas
- **Accuracy**: `(TP + TN) / (TP + TN + FP + FN)`
- **Precision**: `TP / (TP + FP)`
- **Recall (Sensitivity)**: `TP / (TP + FN)`
- **F1-score**: `2 * (Precision * Recall) / (Precision + Recall)`

### 📍 Where to Use Each Metric
- **Accuracy**: Useful when class distribution is balanced.
- **Precision**: Important in scenarios like **spam detection** where false positives must be minimized.
- **Recall**: Critical in **medical diagnosis** where false negatives are costly.
- **F1-score**: Best when **both precision and recall** are equally important.

### 🔀 Problem Statement & Trade-offs
A real-world scenario where focusing on one metric over another can lead to **incorrect conclusions**.

### 🌍 Real-World Examples
- **Email Spam Filtering (Precision vs Recall Trade-off)**
- **Cancer Diagnosis (Importance of Recall Over Precision)**
- **Credit Fraud Detection (Balancing Accuracy and F1-score)**

---

## 🚀 Get Started
Clone this repository and start experimenting with clustering and frequent pattern mining:
```sh
git clone https://github.com/your-username/your-repo.git
cd your-repo
```

Happy coding! 🎉

