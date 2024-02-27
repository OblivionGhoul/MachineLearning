import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report
from sklearn.random_projection import GaussianRandomProjection
from sklearn.decomposition import PCA
from sklearn import cluster

numFeatures = 10

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

print("Label Shape After PCA: ")
print(X.shape)
print("Classification Report: ")
print(classification_report(Y_test, Y_pred))


# PCA
pca = PCA(n_components=numFeatures)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_pca = pca.fit_transform(X_scaled)
print("Label Shape After PCA: ")
print(X_pca.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X_pca, Y, test_size=0.2)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
print("Classification Report with PCA: ")
print(classification_report(Y_test, Y_pred))


# GaussianRandomProjection
rp = GaussianRandomProjection(n_components=numFeatures)
X_rp = rp.fit_transform(X)
print("Label Shape After Gaussian Random Projection: ")
print(X_rp.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X_rp, Y, test_size=0.2)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
print("Classification Report with Gaussian Random Projection: ")
print(classification_report(Y_test, Y_pred))


# FeatureAgglomeration
fa = cluster.FeatureAgglomeration(n_clusters=numFeatures)
X_fa = fa.fit_transform(X)
print("Label Shape After Feature Agglomeration: ")
print(X_fa.shape)
X_train, X_test, Y_train, Y_test = train_test_split(X_fa, Y, test_size=0.2)
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, Y_train)

Y_pred = clf.predict(X_test)
print("Classification Report with Feature Agglomeration: ")
print(classification_report(Y_test, Y_pred))
