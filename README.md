# 📌 Breast Cancer Classification with Logistic Regression (from Scratch)

## 📄 Abstract / Overview

This project implements a binary classification model using **Logistic Regression from scratch** in NumPy. The goal is to predict whether a tumor is **Malignant (M)** or **Benign (B)** using the Wisconsin Breast Cancer dataset.

Unlike off-the-shelf scikit-learn implementations, this project derives and codes the mathematical foundations of logistic regression, including forward propagation, cost calculation, gradient descent, and predictions.

---

## 🏗 Methodology

### Preprocessing
- Removed irrelevant columns (`id`, `Unnamed: 32`)
- Encoded target labels (M → 1, B → 0)
- Min-max normalization applied to features

### Model
- Logistic regression implemented from scratch
- Gradient descent optimization for weights and bias
- Sigmoid activation function
- Binary cross-entropy cost function

### Training
- Manual loop with cost tracking every 100 iterations
- Adjustable learning rate and number of iterations

---

## 📊 Results

- Achieved **~95–97% accuracy** on test set (depending on hyperparameters)
- Convergence observed with decreasing cost function over iterations

---

## 🔬 Reproducibility
```bash
# Clone the repository
git clone https://github.com/yourusername/breast-cancer-logistic-regression-scratch.git
cd breast-cancer-logistic-regression-scratch

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run the model
python main.py
```

### Dependencies (`requirements.txt`)
```text
numpy
pandas
matplotlib
scikit-learn
```

---

## 📈 Visualizations

- Training cost vs iterations plot
- Accuracy comparison on train vs test set

---

## 🚀 Future Work

- Extend to multiclass logistic regression (softmax)
- Compare against scikit-learn `LogisticRegression` baseline
- Add regularization (L2) for improved generalization

---

## 📜 License

[Add your license here, e.g., MIT License]

## 👤 Author

[Your Name]  
[Your GitHub Profile / Contact]
