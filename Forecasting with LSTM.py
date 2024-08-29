import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
data = pd.read_csv(r"/Users/sabarish/Downloads/country_wise_latest.csv")
data.head()
data.isnull().sum()
data.describe()
data.info()
X = data[['Recovered']]
Y = data['New recovered']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, Y_train)
Y_pred = model.predict(X_test)
plt.scatter(X_test, Y_test, color='black', label='Actual data')
plt.plot(X_test, Y_pred, color='blue', linewidth=2, label='Predicted line')
plt.xlabel('Recovered')
plt.ylabel('New recovered')
plt.title('Linear Regression for COVID-19 Data')
plt.legend()
plt.show()
mse = mean_squared_error(X_test, Y_pred)
r2 = r2_score(X_test, Y_pred)
n = len(X_test)
p = X_test.shape[1]
adjusted_r2 = 1 - (1 - r2) * ((n - 1) / (n - p - 1))
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
print(f"Adjusted R^2 Score: {adjusted_r2}")


#charts 
plt.figure(figsize=(6, 4))
plt.bar(['Adjusted R^2'], [adjusted_r2], color='blue')
plt.ylim(0, 1)
plt.title('Adjusted R^2 Score')
plt.ylabel('Score')
plt.show()
