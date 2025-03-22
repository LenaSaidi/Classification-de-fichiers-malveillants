import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

# Load the data
X_train = joblib.load("X_train.pkl")
X_test = joblib.load("X_test.pkl")
y_train = joblib.load("y_train.pkl")
y_test = joblib.load("y_test.pkl")


neighbors = np.arange(1, 14)
train_accuracies = {}
test_accuracies = {}

for neighbor in neighbors:

	knn = KNeighborsClassifier(n_neighbors=neighbor)

	knn.fit(X_train, y_train)
	# y_pred = knn.predict(X_test)
 
	train_accuracies[neighbor] = knn.score(X_train, y_train)
	test_accuracies[neighbor] = knn.score(X_test, y_test)

print(neighbors, '\n', train_accuracies, '\n', test_accuracies)

#plotting
plt.title('k-NN: Varying Number of Neighbors')
plt.plot(neighbors, test_accuracies.values(), label='Testing Accuracy')
plt.plot(neighbors, train_accuracies.values(), label='Training Accuracy')
plt.legend()
plt.xlabel('Number of Neighbors')
plt.ylabel('Accuracy')
plt.show()

