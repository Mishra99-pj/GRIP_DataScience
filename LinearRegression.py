import pandas as pd

data = pd.read_csv(r"http://bit.ly/w-data")

ip = data.drop(['Scores'], axis=1)
op = data['Scores']

from sklearn.model_selection import train_test_split
Xtr,Xts,Ytr,Yts=train_test_split(ip,op,test_size=0.2)


from sklearn.linear_model import LinearRegression
alg=LinearRegression()
alg.fit(Xtr,Ytr)
m=alg.coef_
c=alg.intercept_
print("m is:",m)
print("c is:",c)

import matplotlib.pyplot as plt
plt.scatter(ip,op,c="blue")
y=m*ip+c
plt.plot(ip,y,c="r")
plt.show()

print("Accuracy:",alg.score(Xts,Yts))

import numpy as np
test=np.array([9.25]).reshape(1,-1)
print("Prediction:",alg.predict(test))
