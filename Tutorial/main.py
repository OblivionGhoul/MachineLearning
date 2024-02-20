from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    iris = datasets.load_iris()
    print(iris)

    print("Flower Features:")
    print(iris.feature_names)

    print("Flower Labels:")
    print(iris.target_names)

    print("Flower Data:")
    print(iris.data)

    print("Flower Labels:")
    print(iris.target)

    X = iris.data  # data = feature
    Y = iris.target  # target = label

    print("Data Rows and Columns:")
    print(X.shape)  # 150 samples of data, 4 features each
    print("Feature Rows and Columns:")
    print(Y.shape)  # 150 labels, 1 column

    clf = RandomForestClassifier()
    # creating random forest from given X (features) and Y (labels)
    clf.fit(X, Y)

    print("Importance of Features:")
    print(clf.feature_importances_)

    print(clf.predict([[5.1, 3.5, 1.4, 0.2]]))

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
    clf.fit(X_train, Y_train)
    print(clf.predict([[5.1, 3.5, 1.4, 0.2]]))
    print(clf.predict_proba([[5.1, 3.5, 1.4, 0.2]]))

    print(clf.predict(X_test))
    print(Y_test)

    print(clf.score(X_test, Y_test))
