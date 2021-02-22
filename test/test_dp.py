import numpy as np
from sklearn import datasets
from sklearn.linear_model import LinearRegression as CentralLinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

from app.algo import Coordinator, Client

diabetes_X, diabetes_y = datasets.load_diabetes(return_X_y=True)

# Use only one feature
diabetes_X = diabetes_X[:, np.newaxis, 2]

# Split the data into training/testing sets
X = diabetes_X[:-20]
y = diabetes_y[:-20]
print(X.shape)
X_test = diabetes_X[-20:]
print(X_test.shape)
y_test = diabetes_y[-20:]

print("Central:")
central = CentralLinearRegression()
central.fit(X, y)
central_pred = central.predict(X_test)
# The coefficients
print('Coefficients: \n', central.coef_)
print('Intercept:', central.intercept_)

# The mean squared error
print('Mean squared error: %.2f'
      % mean_squared_error(y_test, central_pred))
# The coefficient of determination: 1 is perfect prediction
print('Coefficient of determination: %.2f'
      % r2_score(y_test, central_pred))

print("Federated:")

client = Coordinator()
X1 = X[:-211]
print(X1.shape)
y1 = y[:-211]

client2 = Client()
X2 = X[-211:]
print(X2.shape)
y2 = y[-211:]

for eps in range(1, 10):
    xtx, xty = client.local_computation(X1, y1, 8)
    xtx2, xty2 = client2.local_computation(X2, y2, 8)
    global_coefs = client.aggregate_beta([[xtx, xty], [xtx2, xty2]])
    client.set_coefs(global_coefs)
    client2.set_coefs(global_coefs)
    print("DP Eps 8 (" + str(eps) + ")" + str(client.coef_))
    fed_pred = client.predict(X_test)
    plt.plot(X_test, fed_pred, linewidth=1, label="$\epsilon$=" + str(8))


plt.scatter(X_test, y_test, color='black')
plt.rcParams["text.usetex"] = True
plt.plot(X_test, central_pred, color='black', linewidth=3)
print("Non-DP:" + str(central.coef_))
plt.legend()
plt.xticks(())
plt.yticks(())
plt.ylim(np.min(y_test)-10, np.max(y_test)+10)

plt.show()



