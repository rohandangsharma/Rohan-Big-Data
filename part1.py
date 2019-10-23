import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

data = np.loadtxt('violentcrime.csv', delimiter=',', skiprows=1, dtype=None)

label = data[:, 1]
features = data[:, 0:1]
lin = LinearRegression().fit(features, label)

print('The score for our linear model is ' + str(lin.score(features, label)))
slope = lin.coef_
inter = lin.intercept_

print('The slope is ' + str(slope) + ', and the y-int is ' + str(inter))
print('When the year is 2020, the crime is ' + str(lin.predict([[2020]])))
print('When the year is 2050, the crime is ' + str(lin.predict([[2050]])))

y_vals = []
for x in features:
    y_vals.append(slope * x + inter)

plt.figure(num='Violent Crime in the US by year')
plt.title('Violent Crime in the US by Year')
plt.xlabel('Year')
plt.ylabel('Crimes')
plt.grid(True)
plt.plot(features, label, 'ro')
plt.plot(features, y_vals, 'b-')
plt.savefig('graph.png')
plt.show()