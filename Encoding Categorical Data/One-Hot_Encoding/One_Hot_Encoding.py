# Importing essential libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Loading the car dataset
df = pd.read_csv('cars.csv')
df  # Displaying the dataset
# -

# Identifying unique categories in the 'fuel' column
fuel = df['fuel'].unique()
print("Unique fuel categories:", fuel)

# Identifying unique car brands and their count
brands = df['brand'].unique()
n_brands = df['brand'].nunique()
print(f"Number of unique brands: {n_brands}")
print("Brands:", brands)

# Identifying unique categories in the 'owner' column
owner = df['owner'].unique()
print("Unique owner categories:", owner)

# --- One-Hot Encoding Using Pandas ---
# Loading a fresh copy of the dataset
df2 = pd.read_csv('cars.csv')

# Performing One-Hot Encoding on 'fuel' and 'owner' columns
encoded_df = pd.get_dummies(df2, columns=['fuel', 'owner'])
print("Dataset after One-Hot Encoding:")
encoded_df

# To avoid multicollinearity, we drop the first category in each encoded column
encoded_df_no_multicollinearity = pd.get_dummies(df2, columns=['fuel', 'owner'], drop_first=True)
print("Dataset after One-Hot Encoding (avoiding multicollinearity):")
encoded_df_no_multicollinearity

# --- One-Hot Encoding on Zomato Dataset ---
# Loading the Zomato dataset
data = pd.read_csv('zomato.csv', encoding='latin1')

# Analyzing the 'Rating color' column
color = data['Rating color']
print("Unique rating colors:", color.unique())
print("Number of unique rating colors:", color.nunique())
print("Frequency of each rating color:")
print(color.value_counts())

# Performing One-Hot Encoding on 'Rating color' column
encoded_data = pd.get_dummies(data=data, columns=['Rating color'])
print("Zomato dataset after One-Hot Encoding:")
encoded_data

# Avoiding multicollinearity in the Zomato dataset
encoded_data_no_multicollinearity = pd.get_dummies(data=data, columns=['Rating color'], drop_first=True)
print("Zomato dataset after One-Hot Encoding (avoiding multicollinearity):")
encoded_data_no_multicollinearity

# --- Splitting Data into Training and Testing Sets ---
from sklearn.model_selection import train_test_split

# Splitting the car dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(
    df[['fuel', 'owner']],  # Features to encode
    df['selling_price'],    # Target variable
    train_size=0.9,         # 90% training data
    random_state=42         # Ensures reproducibility
)
print("Training features:")
x_train
print("Testing features:")
x_test

# Displaying the target variable for training and testing sets
print("Training target:")
y_train
print("Testing target:")
y_test

# --- Applying One-Hot Encoding Using Scikit-learn ---
from sklearn.preprocessing import OneHotEncoder

# Initializing the encoder with 'drop' set to eliminate one category for multicollinearity
ohe = OneHotEncoder(sparse_output=False, drop='first')

# Fitting and transforming the training set
x_train_encoded = ohe.fit_transform(x_train)
print("Encoded training features:")
x_train_encoded

# Transforming the testing set using the same encoder
x_test_encoded = ohe.transform(x_test)
print("Encoded testing features:")
x_test_encoded

# Confirming the shape of the encoded and original feature sets
print("Shape of encoded training set:", x_train_encoded.shape)
print("Shape of original training set:", x_train.shape)

# --- Handling Many Categories with Frequency-Oriented Approach ---
# Identifying car brands with a low frequency
brands = df['brand'].value_counts()
print("Frequency of each brand:")
print(brands)

# Setting a threshold for frequency
threshold = 100
replace = brands[brands < threshold].index  # Identifying brands with frequency below the threshold

# Replacing low-frequency brands with 'Others' and performing One-Hot Encoding
df['brand_replaced'] = df['brand'].replace(replace, 'Others')
encoded_brands = pd.get_dummies(df['brand_replaced'])
print("Dataset after frequency-based One-Hot Encoding on brands:")
encoded_brands
