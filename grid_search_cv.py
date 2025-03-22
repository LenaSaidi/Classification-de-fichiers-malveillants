from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import joblib


X_train = joblib.load("X_train.pkl")
y_train = joblib.load("y_train.pkl")
X_test = joblib.load("X_test.pkl")
y_test = joblib.load("y_test.pkl")


param_grid = {
    'n_neighbors': [3, 5, 7, 9, 11,12,13, 15],  
    'weights': ['uniform', 'distance'],     
    'metric': ['euclidean', 'manhattan', 'minkowski'],
}


grid_search = GridSearchCV(estimator=KNeighborsClassifier(), 
                           param_grid=param_grid,
                           cv=5,
                           scoring='accuracy',
                           n_jobs=-1,
                           verbose=1)  


grid_search.fit(X_train, y_train)

print("Best parameters found: ", grid_search.best_params_)


