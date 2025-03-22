import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score


# Load the data
X_train = joblib.load("X_train.pkl")
X_test = joblib.load("X_test.pkl")
y_train = joblib.load("y_train.pkl")
y_test = joblib.load("y_test.pkl")

knn = KNeighborsClassifier(n_neighbors=13, weights='distance', metric='manhattan')
cv_scores = cross_val_score(knn, X_train, y_train, cv=5, n_jobs=-1, verbose=1, scoring='accuracy')  
print("Cross-validated scores:", cv_scores)
print("Mean accuracy:", cv_scores.mean())

knn.fit(X_train, y_train)

joblib.dump(knn, "knn_model.pkl")

y_pred = knn.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

print("Modèle sauvegardé")
