from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd

df = pd.read_csv("data.csv")

# Given Features
X = df.drop(columns=["Class variable (0 or 1)."])
# Given Labels
Y = df["Class variable (0 or 1)."]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

classifier = KNeighborsClassifier(n_neighbors=3)
classifier.fit(X_train, Y_train)

Y_pred = classifier.predict(X_test)
print("Classification Report: ")
print(classification_report(Y_test, Y_pred))
print("Confusion Matrix: ")
print(confusion_matrix(Y_test, Y_pred))
print("Accuracy Score: ")
print(accuracy_score(Y_pred, Y_test))
