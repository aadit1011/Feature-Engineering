# Importing required libraries
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Reading and preparing the Social Network Ads dataset
df = pd.read_csv('Social_Network_Ads.csv')

# Selecting specific columns for processing
df = df.iloc[:, 2:]

# Splitting the dataset into training and testing sets
# Features: all columns except 'Purchased' (the target)
# Target: 'Purchased'
x_train, x_test, y_train, y_test = train_test_split(
    df.drop('Purchased', axis=1),  # Features
    df['Purchased'],               # Target
    test_size=0.9,                 # 90% test data, 10% training data
    random_state=42                # Ensures reproducibility
)

# Displaying the shape of the training data
print("Training data shape:", x_train.shape)

# Initializing the StandardScaler to standardize the features
scaler = StandardScaler()

# Fitting the scaler on the training data
# The scaler learns the mean and standard deviation of each feature from the training data
scaler.fit(x_train)

# Transforming the training and test sets using the fitted scaler
# Training set is used for fitting and transforming
# Test set is only transformed based on training set parameters
x_train_scaled = scaler.transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Displaying the mean and standard deviation used for scaling
print("Mean of features in training set:", scaler.mean_)
print("Standard deviation of features in training set:", scaler.scale_)

# Displaying the scaled training and test datasets
print("Standardized training set:\n", x_train_scaled)
print("Standardized test set:\n", x_test_scaled)

# Example: Small synthetic dataset to demonstrate standardization
# Creating a small example dataset
X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
y = np.array([0, 1, 0, 1, 0])

# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initializing the StandardScaler and applying it to the example data
example_scaler = StandardScaler()
X_train_scaled = example_scaler.fit_transform(X_train)  # Fit and transform training set
X_test_scaled = example_scaler.transform(X_test)        # Transform test set only

# Displaying the results for the synthetic dataset
print("Example Dataset:\n")
print("Mean of training set:", example_scaler.mean_)
print("Standard deviation of training set:", example_scaler.scale_)
print("Standardized training set:\n", X_train_scaled)
print("Standardized test set:\n", X_test_scaled)

# Applying standardization on another example dataset (insurance data)
# Loading insurance dataset
df_insurance = pd.read_csv('insurance2.csv')

# Selecting specific columns for feature and target
df_insurance = df_insurance.iloc[:, [0, 2, -2, -1]]  # Selecting columns by index

# Splitting features and labels
x_insurance = df_insurance.iloc[:, :-1]  # Features (all but the last column)
y_insurance = df_insurance.iloc[:, -1]   # Target (last column)

# Splitting the insurance dataset into training and testing sets
x_train_insurance, x_test_insurance, y_train_insurance, y_test_insurance = train_test_split(
    x_insurance, y_insurance, train_size=0.9, random_state=42
)

# Initializing the StandardScaler
insurance_scaler = StandardScaler()

# Fitting the scaler on the training data and transforming both training and test sets
x_train_insurance_scaled = insurance_scaler.fit_transform(x_train_insurance)
x_test_insurance_scaled = insurance_scaler.transform(x_test_insurance)

# Displaying the scaled insurance dataset
print("Insurance Dataset:\n")
print("Mean of features in training set:", insurance_scaler.mean_)
print("Standard deviation of features in training set:", insurance_scaler.scale_)
print("Standardized training set:\n", x_train_insurance_scaled)
print("Standardized test set:\n", x_test_insurance_scaled)
