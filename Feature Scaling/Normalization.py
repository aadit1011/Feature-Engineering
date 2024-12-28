# Importing necessary libraries
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# Load the dataset
# Reading wine data CSV with specified columns and adding appropriate headers.
df = pd.read_csv('wine_data.csv', usecols=[0, 1, 2], header=None, names=['Class Label', 'Alcohol', 'Malic Acid'])

# Display the first few rows of the dataset for verification
print(df.head())

# KDE Plot: Visualizing the distribution of 'Alcohol'
sns.kdeplot(df['Alcohol'])
plt.title('KDE Plot of Alcohol')
plt.savefig('kde_alcohol.png')
plt.show()

# KDE Plot: Visualizing the distribution of 'Malic Acid'
sns.kdeplot(df['Malic Acid'])
plt.title('KDE Plot of Malic Acid')
plt.savefig('kde_malic_acid.png')
plt.show()

# Scatter Plot: Relationship between 'Alcohol' and 'Malic Acid', grouped by 'Class Label'
sns.scatterplot(x=df['Alcohol'], y=df['Malic Acid'], hue=df['Class Label'], palette=['red', 'blue', 'green'])
plt.title('Scatter Plot: Malic Acid vs Alcohol')
plt.savefig('scatter_malic_acid_vs_alcohol.png')
plt.show()

# Splitting the dataset into training and test sets
# Features (X) exclude 'Class Label'; Target (y) is 'Class Label'.
X_train, X_test, y_train, y_test = train_test_split(
    df.drop('Class Label', axis=1), df['Class Label'], test_size=0.1, random_state=0
)
print(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# Scaling features using Min-Max Scaler
scaler = MinMaxScaler()
scaler.fit(X_train)

# Transforming the training and test sets
X_train_scaled = pd.DataFrame(scaler.transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# Displaying the first few rows of scaled training and test sets
print("Scaled Training Set:")
print(X_train_scaled.head())
print("Scaled Test Set:")
print(X_test_scaled.head())

# Comparing descriptive statistics before and after scaling
print("Descriptive statistics before scaling:")
print(np.round(X_train.describe(), 3))

print("Descriptive statistics after scaling:")
print(np.round(X_train_scaled.describe(), 3))

# Scatter Plot: Visualizing the effect of scaling
# Before scaling
sns.scatterplot(x=X_train['Alcohol'], y=X_train['Malic Acid'], hue=y_train)
plt.title('Scatter Plot Before Scaling')
plt.savefig('scatter_before_scaling.png')
plt.show()

# After scaling
sns.scatterplot(x=X_train_scaled['Alcohol'], y=X_train_scaled['Malic Acid'], hue=y_train)
plt.title('Scatter Plot After Scaling')
plt.savefig('scatter_after_scaling.png')
plt.show()

# KDE Plot: Visualizing distributions before and after scaling
# Before scaling
sns.kdeplot(x=X_train['Alcohol'], label='Alcohol (Before Scaling)')
sns.kdeplot(x=X_train['Malic Acid'], label='Malic Acid (Before Scaling)')
plt.title('KDE Plot Before Scaling')
plt.legend()
plt.savefig('kde_before_scaling.png')
plt.show()

# After scaling
sns.kdeplot(x=X_train_scaled['Alcohol'], label='Alcohol (After Scaling)')
sns.kdeplot(x=X_train_scaled['Malic Acid'], label='Malic Acid (After Scaling)')
plt.title('KDE Plot After Scaling')
plt.legend()
plt.savefig('kde_after_scaling.png')
plt.show()