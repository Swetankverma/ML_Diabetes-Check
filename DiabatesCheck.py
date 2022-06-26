from tkinter import image_names
import matplotlib.pyplot as plt
import numpy as np
from sklearn  import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes = datasets.load_diabetes()

diabetes_X = diabetes.data

print(diabetes_X)

diabetes_X_train = diabetes_X[:-30]
diabetes_X_test = diabetes_X[-30:]

diabetes_y_train = diabetes.target[:-30]
diabetes_y_test = diabetes.target[-30:]

model = linear_model.LinearRegression()

model.fit(diabetes_X_train, diabetes_y_train)

diabetes_y_predicated = model.predict(diabetes_X_test)

print("Mean squared error is :", mean_squared_error(diabetes_y_test, diabetes_y_predicated))

print("weight", model.coef_)
print("Intercept", model.intercept_)

#plt.scatter(diabetes_X_test, diabetes_y_test)
#plt.plot(diabetes_X_test, diabetes_y_predicated)

#plt.show()

#Mean squared error is : 3035.060115291269
#weight [941.43097333]d
#Intercept 153.39713623331644
