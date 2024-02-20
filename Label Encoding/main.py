import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Read the CSV file
df = pd.read_csv("UNR-IDD.csv")
le = LabelEncoder()

# Given Features
df["Switch ID"] = le.fit_transform(df["Switch ID"])
df["Port Number"] = le.fit_transform(df["Port Number"])
df["is_valid"] = le.fit_transform(df["is_valid"])
X = df.drop(columns=["Label", "Binary Label"])

# Given Labels
df["Binary Label"] = le.fit_transform(df["Binary Label"])
df["Label"] = le.fit_transform(df["Label"])
Y = df["Label"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)

print("Classification Report: ")
print(classification_report(Y_test, Y_pred))
print("Confusion Matrix: ")
print(confusion_matrix(Y_test, Y_pred))
print("Accuracy Score: ")
print(accuracy_score(Y_pred, Y_test))
