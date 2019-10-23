import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = np.loadtxt('jaxWeather.csv', delimiter=',', skiprows=1, dtype=None)

label = data[:, 1]
features = data[:, 1:3]
features_train, features_test, label_train, label_test = train_test_split(features, label)

lin = LinearRegression().fit(features_train, label_train)
slope = lin.coef_
inter = lin.intercept_

print('the training score is: ' + str(lin.score(features_train, label_train)))
print('the testing score is: ' + str(lin.score(features_test, label_test)))

fig = plt.figure()
ax = Axes3D(fig)

start = 1
end = start + 365
print(str(start) + ' ' + str(end))

ax.scatter(data[start:end, 0], data[start:end, 1], data[start:end, 2])
# ax.plot(features[0], features[2], lin.predict(np.column_stack((features[0], features[2]) ) ), 'r-')

# features_train_poly = PolynomialFeatures(2).fit_transform(features_train)
# features_test_poly = PolynomialFeatures(2).fit_transform(features_test)
#
# feature_names = []
# for x in range(0, len(features[0]) ):
#     feature_names.append('feature' + str(x + 1))
# poly_feature_names = PolynomialFeatures(2).fit(features_train).get_feature_names(feature_names)

ax.set_xlabel('Date')
ax.set_ylabel('MinTemp')
ax.set_zlabel('MaxTemp')

plt.show()
