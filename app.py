# ====================================================
# Import Libraries
# ====================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# ====================================================
# Load and Preprocess Data
# ====================================================
data = pd.read_csv("data.csv")
print(data.head())  # Show first 5 rows

# Display dataset info
data.info()

# Drop irrelevant columns
data.drop(['Unnamed: 32', 'id'], axis=1, inplace=True)

# Encode diagnosis column: M -> 1 (Malignant), B -> 0 (Benign)
data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})

# Separate features and labels
y = data['diagnosis'].values
x_data = data.drop(['diagnosis'], axis=1)

# Normalize feature data
x = (x_data - x_data.min()) / (x_data.max() - x_data.min())


# ====================================================
# Split into Train/Test Sets
# ====================================================
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.15, random_state=42
)

# Transpose for easier matrix operations
x_train, x_test = x_train.T, x_test.T
y_train, y_test = y_train.T, y_test.T

print("x_train:", x_train.shape)
print("x_test:", x_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# ====================================================
# Helper Functions
# ====================================================
def initialize_weights_and_bias(dimension):
    """Initialize weights (small random values) and bias (zero)."""
    w = np.random.randn(dimension, 1) * 0.01  
    b = 0.0
    return w, b


def sigmoid(z):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-z))


def forward_backward_propagation(w, b, x_train, y_train):
    """Perform forward and backward propagation and compute gradients."""
    m = x_train.shape[1]

    # Forward propagation
    z = np.dot(w.T, x_train) + b
    y_head = sigmoid(z)

    # Cost function
    cost = (-1 / m) * np.sum(
        y_train * np.log(y_head) + (1 - y_train) * np.log(1 - y_head)
    )

    # Backward propagation (gradients)
    derivative_weight = (1 / m) * np.dot(x_train, (y_head - y_train).T)
    derivative_bias = (1 / m) * np.sum(y_head - y_train)

    gradients = {"derivative_weight": derivative_weight, "derivative_bias": derivative_bias}
    return cost, gradients


def update(w, b, x_train, y_train, learning_rate, num_iterations):
    """Gradient descent update of weights and bias."""
    costs = []

    for i in range(num_iterations):
        cost, grad = forward_backward_propagation(w, b, x_train, y_train)

        # Update weights and bias
        w -= learning_rate * grad["derivative_weight"]
        b -= learning_rate * grad["derivative_bias"]

        # Track cost every 100 iterations
        if i % 100 == 0:
            costs.append(cost)
            print(f"Cost after iteration {i}: {cost}")

    parameters = {"weight": w, "bias": b}
    return parameters, grad, costs


def predict(w, b, x_test):
    """Predict labels for test data using learned weights and bias."""
    m = x_test.shape[1]
    y_prediction = np.zeros((1, m))
    z = sigmoid(np.dot(w.T, x_test) + b)

    # Apply threshold at 0.5
    for i in range(z.shape[1]):
        y_prediction[0, i] = 1 if z[0, i] > 0.5 else 0

    return y_prediction


# ====================================================
# Logistic Regression Model
# ====================================================
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate=0.01, num_iterations=1000):
    """Train logistic regression model and evaluate accuracy."""
    dimension = x_train.shape[0]

    # Initialize parameters
    w, b = initialize_weights_and_bias(dimension)

    # Train using gradient descent
    parameters, gradients, costs = update(w, b, x_train, y_train, learning_rate, num_iterations)

    # Predictions
    y_prediction_test = predict(parameters["weight"], parameters["bias"], x_test)
    y_prediction_train = predict(parameters["weight"], parameters["bias"], x_train)

    # Accuracy
    print(f"Train accuracy: {100 - np.mean(np.abs(y_prediction_train - y_train)) * 100:.2f}%")
    print(f"Test accuracy: {100 - np.mean(np.abs(y_prediction_test - y_test)) * 100:.2f}%")


# ====================================================
# Run the Model
# ====================================================
logistic_regression(x_train, y_train, x_test, y_test, learning_rate=0.01, num_iterations=1000)
