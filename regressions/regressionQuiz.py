import numpy
import matplotlib.pyplot as plt

from ages_net_worths import ageNetWorthData

ages_train, ages_test, net_worths_train, net_worths_test = ageNetWorthData()



from sklearn.linear_model import LinearRegression

reg = LinearRegression()
reg.fit(ages_train, net_worths_train)

### get Katie's net worth (she's 27)
print "Katie's net worth: ", reg.predict([[27]])[0][0]
# The coefficients
print "Coefficient: ", reg.coef_
# The intercept
print "Intercept: ", reg.intercept_
# The r-squared on test data
print "R-squared on test data: ", reg.score(ages_test, net_worths_test)
# The r-squared on training data
print "R-squared on training data: ", reg.score(ages_train, net_worths_train)
# The mean squared error
print "Mean squared error: ", ((reg.predict(ages_test) - net_worths_test) ** 2)
# Explained variance score: 1 is perfect prediction
print "Variance score: ", reg.score(ages_test, net_worths_test)
### sklearn predictions are returned in an array, so you'll want to index into
### the output to get what you want, e.g. net_worth = predict([[27]])[0][0] (not
### exact syntax, the point is the [0] at the end). In addition, make sure the
### argument to your prediction function is in the expected format - if you get
### a warning about needing a 2d array for your data, a list of lists will be
### interpreted by sklearn as such (e.g. [[27]]).
km_net_worth = 160.432054531 ### fill in the line of code to get the right value

### get the slope
### again, you'll get a 2-D array, so stick the [0][0] at the end
slope = 6.47354955 ### fill in the line of code to get the right value

### get the intercept
### here you get a 1-D array, so stick [0] on the end to access
### the info we want
intercept = -14.35378331 ### fill in the line of code to get the right value

### get the score on test data
test_score = 0.812365729231 ### fill in the line of code to get the right value

### get the score on the training data
training_score = 0.874588235822 ### fill in the line of code to get the right value



def submitFit():
    # all of the values in the returned dictionary are expected to be
    # numbers for the purpose of the grader.
    return {"networth":km_net_worth,
            "slope":slope,
            "intercept":intercept,
            "stats on test":test_score,
            "stats on training": training_score}