import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data = np.loadtxt('NFLRB.csv', delimiter = ',', skiprows=1, usecols=(1,2,3), dtype=None)
# where your code should go that imports the data (features, label) and then splits it into testing and training

label = data[ : , 1]
features = data[ : , 1:4]
features_train, features_test, label_train, label_test = train_test_split(features, label)

lin = LinearRegression().fit(features_train, label_train)
slope = lin.coef_
inter = lin.intercept_

print ('the training score is: ' + str(lin.score(features, label) ) )
print ('the training score is: ' + str(lin.score(features_train, label_train) ) )
print ('the testing score is: ' + str(lin.score(features_test, label_test) ) )

features_train_poly = PolynomialFeatures(2).fit_transform(features_train)
features_test_poly = PolynomialFeatures(2).fit_transform(features_test)

feature_names = []
for x in range(0, len(features[0]) ):
    feature_names.append('feature' + str(x + 1))
poly_feature_names = PolynomialFeatures(2).fit(features_train).get_feature_names(feature_names)
print (poly_feature_names)

lin = LinearRegression().fit(features_train_poly, label_train)

print ('the training score of our linear model is ' + str( lin.score(features_train_poly, label_train) ) )
print ('the testing score is: ' + str(lin.score(features_test_poly, label_test) ) )

string = 'output = ' + str(lin.intercept_)
for x in range(0, len( lin.coef_ ) ):
    string += ' + ' + str(lin.coef_[x]) + ' * ' + poly_feature_names[x]
print (string)
