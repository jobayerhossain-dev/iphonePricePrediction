# Import required libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Read the CSV data into DataFrame
data = pd.read_csv('iphone_price.csv')

# Create a scatter plot of version vs price
plt.scatter(data['version'], data['price'])
plt.xlabel('IPhone Version')
plt.ylabel('Price')
plt.title('IPhone Price Distribution')  # Set a title for the plot
plt.savefig('iphone_price.png')  # Save the plot as an image

# Initialize a LinearRegression model & Data Train
model = LinearRegression()
X = data[['version']]
y = data['price']
model.fit(X, y)

# Predict the price for a given version (e.g. version 16)
predicted_price = model.predict([[16]])
print(predicted_price)
