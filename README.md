# 📌 Breast Cancer Classification with Logistic Regression (from Scratch)

This project implements a binary classification model using **Logistic Regression from scratch** in NumPy. The goal is to predict whether a tumor is Malignant (M) or Benign (B) using the Wisconsin Breast Cancer dataset.

Unlike off-the-shelf `scikit-learn` implementations, this project derives and codes the mathematical foundations of logistic regression, including forward propagation, cost calculation, gradient descent, and predictions.

---

## 🏗 Methodology

The model is built from the ground up, focusing on the core mathematical components.

### Preprocessing
* Removed irrelevant columns (`id`, `Unnamed: 32`).
* Encoded target labels (Malignant $\rightarrow$ 1, Benign $\rightarrow$ 0).
* Applied Min-Max normalization to scale all features between 0 and 1.

### Model
* **Logistic Regression:** Implemented from scratch using only NumPy.
* **Activation:** Sigmoid function ($\sigma(z) = \frac{1}{1 + e^{-z}}$).
* **Cost Function:** Binary Cross-Entropy (Log Loss).
* **Optimization:** Gradient Descent to update weights ($w$) and bias ($b$).

### Training
* A manual training loop is used, with an adjustable learning rate and number of iterations.
* Cost is tracked and printed every 100 iterations to monitor convergence.

---

## 📊 Results

* Achieved **~95–97% accuracy** on the test set, depending on hyperparameter tuning.
* Observed successful convergence with the cost function decreasing steadily over iterations.

---

## 🔬 Getting Started

Follow these steps to download and run the project locally.

### 1. Clone the Repository
```bash
git clone [https://github.com/yourusername/breast-cancer-logistic-regression-scratch.git](https://github.com/yourusername/breast-cancer-logistic-regression-scratch.git)
cd breast-cancer-logistic-regression-scratch
