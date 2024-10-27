import numpy as np
from sklearn import linear_model
import sklearn.metrics as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

m = 100
X = 6 * np.random.rand(m, 1) - 4
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

# Побудова графіка Лінійна регресія
fig, ax = plt.subplots()
ax.scatter(X, y, color='green')
plt.title("Лінійна регресія")
plt.show()

print(X[1], y[1])

poly_features = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly_features.fit_transform(np.array(X).reshape(-1, 1))

linear_regression = linear_model.LinearRegression()
linear_regression.fit(X_poly, y)
print(linear_regression.intercept_, linear_regression.coef_)
y_pred = linear_regression.predict(X_poly)
print("\nR2 = ", sm.r2_score(y, y_pred))
fig, ax = plt.subplots()
ax.scatter(X, y, color='green')
plt.plot(X, y_pred, "*", color='black', linewidth=4)
plt.title("Поліноміальна регресія")
plt.show()
