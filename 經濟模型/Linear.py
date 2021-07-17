import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn import datasets
def mod1():
    X,y=datasets.make_regression(n_samples=100, n_features=1, noise=50)
    X = np.interp(X, (X.min(), X.max()), (1, 20))

    y = np.interp(y, (y.min(), y.max()), (1, 20))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    plt.scatter(X,y)
    print(X)
    regr=linear_model.LinearRegression()
    regr.fit(X_train, y_train)


    plt.scatter(X_train, y_train, color='black')

    plt.plot(X_train, regr.predict(X_train),color='blue',linewidth=1)
    plt.show()

    w_0=regr.intercept_
    w_1=regr.coef_


    plt.scatter(X_test, y_test, color='red')
    plt.plot(X_test, regr.predict(X_test),color='blue',linewidth=1)
    plt.show()
    print("訓練集的分數:",regr.score(X_train,y_train))
    print("測試集的分數:",regr.score(X_test,y_test))
    print(w_0,w_1)
    return w_0,w_1
def mod2():
    X,y=datasets.make_regression(n_samples=100, n_features=1, noise=50)
    X = np.interp(X, (X.min(), X.max()), (20, 1))
    
    y = np.interp(y, (y.min(), y.max()), (1, 20))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    plt.scatter(X,y)
    print(X)
    regr=linear_model.LinearRegression()
    
    regr.fit(X_train, y_train)


    plt.scatter(X_train, y_train, color='black')

    plt.plot(X_train, regr.predict(X_train),color='blue',linewidth=1)
    plt.show()

    w_0=regr.intercept_
    w_1=regr.coef_


    plt.scatter(X_test, y_test, color='red')
    plt.plot(X_test, regr.predict(X_test),color='blue',linewidth=1)
    plt.show()
    print("訓練集的分數:",regr.score(X_train,y_train))
    print("測試集的分數:",regr.score(X_test,y_test))
    print(w_0,w_1)
    return w_0,w_1
