# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Preparation: Load dataset, handle missing values, encode categorical variables, and separate features (X) and target (y).

2. Data Splitting: Split data into training and testing sets using train_test_split with an 80-20 ratio.

3. Model Training: Initialize and train a DecisionTreeRegressor with the training data.

4. Evaluation: Predict on test data, calculate mean squared error and R-squared score, and test model predictions.


## Program and Outputs:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: SETHUKKARASI C
RegisterNumber:  212223230201
*/
```
<br>

```
import pandas as pd
```
<br>

```
data = pd.read_csv("Salary.csv")
```
<br>

```
data.head()
```
<br>

![out1](/o1.png)
<br>

```
data.info()
```
<br>

![out2](/o2.png)
<br>

```
data.isnull().sum()
```
<br>

![out3](/o3.png)
<br>

```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
```
<br>

```
data["Position"] = le.fit_transform(data["Position"])
data.head()
```
<br>

![out4](/o4.png)
<br>

```
x = data[["Position", "Level"]]
y = data["Salary"]
```
<br>

```
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state = 11)
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(X_train, Y_train)
```
<br>

![out5](/o5.png)
<br>

```
Y_pred = dt.predict(X_test)
print(Y_pred)
```
<br>

![out6](/o6.png)
<br>

```
from sklearn.metrics import mean_squared_error,r2_score
mse = mean_squared_error(Y_test, Y_pred)
print(mse)
```
<br>

![out7](/o7.png)
<br>

```
r2=r2_score(Y_test, Y_pred)
print(r2)
```
<br>

![out8](/o8.png)
<br>

```
dt.predict([[5,6]])
```
<br>

![out9](/o9.png)
<br>

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
