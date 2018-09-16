# stock price predictor

# importing all important stuff
import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# display graph backend(change it required)
plt.switch_backend('GTK3Agg')

dates = []
prices = []

# adding data
def get_data(filename):
    with open(filename, 'r') as csvfile:
        csvFileReader = csv.reader(csvfile)
        next(csvFileReader)
        for row in csvFileReader:
            print(row)
            dates.append(int(row[0].split('/')[2]))
            prices.append(float(row[1]))
    return 

# training model with data
def predict_prices(dates, prices, x):
    dates = dates[:x+1]
    prices = prices[:x+1]
    dates = np.reshape(dates, (len(dates), 1))

    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_lin.fit(dates, prices)
    svr_poly.fit(dates, prices)
    svr_rbf.fit(dates, prices)

    # adding predictions to graph
    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF model')
    plt.plot(dates, svr_lin.predict(dates), color='blue', label='Linear model')
    plt.plot(dates, svr_poly.predict(dates), color='red', label='Polynomial model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()

    return svr_rbf.predict(x)[0], svr_lin.predict(x)[0], svr_poly.predict(x)[0]

get_data('aapl.csv')

predicted_prices = predict_prices(dates, prices, 29)

print(predicted_prices)