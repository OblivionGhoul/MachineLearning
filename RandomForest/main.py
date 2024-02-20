from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
df = pd.read_csv("data.csv")

# Given Features
X = df.drop(columns=["Class variable (0 or 1)."])
# Given Labels
Y = df["Class variable (0 or 1)."]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, Y_train)

print("Predicting Labels: ")
print(clf.predict(X_test))

print("Actual Labels: ")
print(Y_test)

print("Accuracy Score: ")
print(clf.score(X_test, Y_test))
