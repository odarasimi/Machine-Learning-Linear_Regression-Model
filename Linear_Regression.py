import pandas as pd
import quandl
import math
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle
from statistics import mean
import matplotlib.pyplot as plt
from matplotlib import style
import random

df = pd.read_csv("datasets_WikiGoogl.csv")
'''
#sets desired data for the dataset
df = df[["Adj. Open", "Adj. High", "Adj. Low", "Adj. Close", "Adj. Volume"]]
df["HL_PCT"] = (df["Adj. High"] - df["Adj. Close"]) / df["Adj. Close"] * 100.0
df["PCT_change"] = (df["Adj. Close"] - df["Adj. Open"]) / df["Adj. Open"] * 100.0
df = df[["Adj. Open", "Adj. Close", "HL_PCT", "PCT_change", "Adj. Volume"]]

forecast_col = "Adj. Close"
df.fillna(-99999, inplace = True)
forecast_out = int(math.ceil(0.01*len(df)))

# sets the output data for training
df["Label"] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)

X = np.array(df.drop(["Label"],1))
y = np.array(df["Label"])
X = preprocessing.scale(X)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size = 0.2)

cvr = LinearRegression()
cvr.fit(X_train, y_train)

#saves the optimisation process into a file*******************
with open ("LinearRegression.pickle", "wb") as md:
	pickle.dump(cvr, md)
pickle_in = open("LinearRegression.pickle", "rb")
#*************************************************************
cvr = pickle.load(pickle_in)

print (cvr.score(X_test,y_test))
'''
style.use("fivethirtyeight")
xs = (np.array([1, 2, 3, 4, 5, 6]))
ys = (np.array([100000, 200000, 250000, 320000, 550000, 700000]))

#returns variables of the regression line equation using the least square method
def best_fit_line(x, y):
	m = (mean(xs) * mean(ys) - mean(xs*ys)) / (mean(xs)**2 - mean(xs**2))
	b = mean(ys) - m*mean(xs)
	return m, b

#this is a measure of how accurate the regression line is using the r error method
def squared_error(ys_orig, ys_line):
	return sum((ys_orig-ys_line)**2)

def coefficient_of_determination(ys_orig, ys_line):
	ys_mean_line = [mean(ys_orig) for y in ys_orig]
	squared_error_regr = squared_error(ys_orig, ys_line)
	squared_error_mean = squared_error(ys_orig, ys_mean_line)
	return (1 - (squared_error_regr/squared_error_mean))

#creates the regression line
m, b = best_fit_line(xs, ys)
regression_line = [(m*x)+b for x in xs]





