Breast Cancer Classification with Logistic Regression (from Scratch)
This project builds a binary classifier in NumPy to predict Malignant (1) vs Benign (0) tumors using the Wisconsin Breast Cancer dataset. It implements the math and training loop of logistic regression without scikit-learn’s model APIs.

Methodology
Preprocessing

Dropped non-informative columns: id, Unnamed: 32

Encoded labels: M → 1, B → 0

Applied min–max normalization on features

Model

Logistic regression with sigmoid activation

Binary cross-entropy loss

Gradient descent to learn weights and bias

Training

Iterative updates with adjustable learning rate and iterations

Cost tracked every 100 steps

Train/test split via scikit-learn utilities

Results
Test accuracy: ~95–97% (varies with hyperparameters)

Monotonic decrease in cost indicates good convergence

Visualizations
Training cost vs iterations

Accuracy comparison: train vs test

Reproducibility
Clone repository

text
git clone <your-repo-url>
cd breast-cancer-logistic-regression-scratch
Create virtual environment

macOS/Linux

text
python -m venv venv
source venv/bin/activate
Windows

text
python -m venv venv
venv\Scripts\activate
Install dependencies

text
pip install -r requirements.txt
Run

text
python main.py
Requirements
text
numpy
pandas
matplotlib
scikit-learn
Project Structure
data/: dataset CSVs or download script

src/: core implementation (forward pass, loss, gradients, training loop)

plots/: generated figures

main.py: entry point

requirements.txt: dependencies

README.md: documentation

Future Work
Add L2 regularization to improve generalization

Extend to softmax for multiclass

Compare against scikit-learn’s LogisticRegression baseline
