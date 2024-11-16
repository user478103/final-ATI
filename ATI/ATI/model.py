import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

boston = {
    'data': data,
    'target': target,
}
num_samples, num_features = boston['data'].shape
print(f"The dataset contains {num_samples} samples and {num_features} features.")
df = pd.DataFrame(boston['data'])
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT']
df['PRICE'] = boston['target']
df.head()
df.describe()
sns.distplot(df['PRICE'])
df.corr()
sns.lmplot(x='CRIM', y='PRICE', data=df)
sns.lmplot(x='RM', y='PRICE', data=df)
sns.lmplot(x='TAX', y='PRICE', data=df)
from sklearn.preprocessing import StandardScaler

# Assuming X is feature matrix (e.g., X = df[['CRIM', 'ZN', 'INDUS', ...]])

# Create a StandardScaler object
scaler = StandardScaler()
X = df.drop('PRICE', axis=1).values  # Features
y = df['PRICE'].values  # Target

# Standardize the feature matrix
X_scaled = scaler.fit_transform(X)
# Add a bias column (intercept term) to the feature matrix
X_scaled = np.c_[np.ones(X_scaled.shape[0]), X_scaled]  # Add a column of ones for the intercept term
from sklearn.model_selection import train_test_split # Import train_test_split

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
# Function to compute the cost (Mean Squared Error)
def compute_cost(X, y, theta):
    m = len(y)
    predictions = X.dot(theta)
    cost = (1 / (2 * m)) * np.sum((predictions - y) ** 2)
    return cost

# Function to perform gradient descent
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, len(theta))) # To store theta values after each iteration

    for i in range(iterations):
        # Calculate the prediction error
        predictions = X.dot(theta)

        # Calculate the gradient
        gradient = (1 / m) * X.T.dot(predictions - y)

        # Update the parameters (coefficients)
        theta -= learning_rate * gradient

        # Store the cost for this iteration
        cost_history[i] = compute_cost(X, y, theta)
        theta_history[i, :] = theta

        # Print theta values
        print(f"Iteration {i+1}: Theta = {theta}, Cost = {cost_history[i]}")

    return theta, cost_history

# Initialize theta (coefficients) with zeros
theta_initial = np.zeros(X_train.shape[1])

# Set hyperparameters: learning rate and number of iterations
learning_rate = 0.01
iterations = 200

# Perform gradient descent to find the best coefficients
theta_optimal, cost_history = gradient_descent(X_train, y_train, theta_initial, learning_rate, iterations)

# Print the optimal coefficients (theta values)
print("Optimal coefficients:", theta_optimal)
# Make predictions on the test set
y_pred = X_test.dot(theta_optimal)
pd.DataFrame({"Actual Price": y_test, "Predicted Price": y_pred})

# Function to predict house price based on user input
def predict_house_price(input_features):
    # Standardize the input features using the same scaler used during training
    input_scaled = scaler.transform([input_features])

    # Add a column of ones to account for the intercept term
    input_scaled = np.c_[np.ones(input_scaled.shape[0]), input_scaled]

    # Predict the price using the optimal theta (trained model)
    predicted_price = input_scaled.dot(theta_optimal)

    return predicted_price[0]

