import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as snb
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv('P1_diamonds.csv')

#print(df.head(10).to_string())

#delete unnamed column
df = df.drop(['Unnamed: 0'], axis=1)

#create variables for categories
categorical_features = ['cut', 'color', 'clarity']
le = LabelEncoder()

#replace categories by numbers
for i in range(3):
    new = le.fit_transform(df[categorical_features[i]])
    df[categorical_features[i]] = new
    
#print(df.head(10).to_string())

X = df[['carat', 'cut', 'color', 'clarity', 'depth', 'table', 'x','y', 'z']]
y = df[['price']]

#split data to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=25, random_state=101)

#train
regr = RandomForestRegressor(n_estimators=10, max_depth=10, random_state=101)
regr.fit(X_train, y_train.values.ravel())

#prediction
predictions = regr.predict(X_test)

result = X_test
result['price'] = y_test
result['prediction'] = predictions.tolist()

print(result.to_string())

#define x axis
x_axis = X_test.carat

#plotting
plt.scatter(x_axis, y_test, c='b', alpha=0.5, marker='.', label='Real')
plt.scatter(x_axis, predictions, c='r', alpha=0.5, marker='.', label='Predicted')
plt.xlabel('Carat')
plt.ylabel('Price')
plt.grid(color='#D3D3D3', linestyle='solid')
plt.legend(loc='lower right')
plt.show()






