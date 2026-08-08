# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries such as NumPy, Pandas, Matplotlib, and the SGDRegressor model from Scikit-learn.

2. Load and preprocess the dataset by separating the input features (independent variables) and the target variables (house price and number of occupants). Split the dataset into training and testing sets.

3. Train the model by creating separate SGDRegressor models for each target variable (house price and number of occupants) and fit them using the training data.

4 .Predict and evaluate the results using the testing data, compare the predicted values with the actual values, and calculate performance metrics such as Mean Squared Error (MSE) and R² Score.
## Program:
```
/*
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Udhaya.S
RegisterNumber: 212225230287 
*/
```
~~~ python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
#load california housing dataset
data=fetch_california_housing()
X=data.data
y = data.target.reshape(-1, 1)
# Scale input features
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
# Scale target
scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y)
# Add bias column
X_scaled = np.c_[ np.ones(X_scaled.shape[0]),X_scaled]

# SGD Linear Regression
def linear_regression_sgd(X, y, learning_rate=0.001, epochs=100):
    m, n = X.shape

    # Initialize theta
    theta = np.zeros((n, 1))

    for epoch in range(epochs):

        # Shuffle data
        indices = np.random.permutation(m)

        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # Update theta for each sample
        for i in range(m):

            xi = X_shuffled[i:i+1]
            yi = y_shuffled[i:i+1]

            # Prediction
            prediction = xi.dot(theta)

            # Error
            error = prediction - yi

            # Gradient
            gradient = xi.T.dot(error)

            # Update theta
            theta = theta - learning_rate * gradient

    return theta
# Train model
theta = linear_regression_sgd(X_scaled,y_scaled,learning_rate=0.001,epochs=100)
print("Theta:")
print(theta)
new_data = np.array([[8.3252,41.0,6.984127,1.023810,322.0,2.555556,37.88,-122.23]])
# Scale new data
new_scaled = scaler_X.transform(new_data)
# Add bias
new_scaled = np.c_[np.ones(new_scaled.shape[0]),new_scaled]
# Scale new data
new_scaled = scaler_X.transform(new_data)
# Add bias
new_scaled = np.c_[np.ones(new_scaled.shape[0]),new_scaled]
~~~

## Output:
<img width="323" height="230" alt="image" src="https://github.com/user-attachments/assets/61352d90-abea-4c8f-b24f-58ceea2a1209" />

<img width="610" height="151" alt="Screenshot 2026-08-08 142512" src="https://github.com/user-attachments/assets/3f731c37-5e19-4eeb-a0b6-ff1d167087b2" />

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
