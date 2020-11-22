import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

time_studied = np.array([20, 50, 32, 65, 23, 43, 10, 5, 22, 35, 29, 5, 56]).reshape(-1, 1) #Data for time studied
scores = np.array([56, 83, 47, 93, 47, 82, 45, 78, 55, 67, 57, 4, 12]).reshape(-1, 1) #Data for test scores

model = LinearRegression()
model.fit(time_studied, scores)

print(model.predict(np.array([56]).reshape(-1, 1))) #Prints estimate based on data

#Prints the chart
plt.scatter(time_studied, scores)
plt.plot(np.linspace(0, 70, 100), model.predict(np.linspace(0, 70, 100).reshape(-1, 1)), 'r')
plt.ylim(0, 100)
plt.show()