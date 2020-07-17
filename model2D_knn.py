from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import model2D_preprocessing as pp
import matplotlib.pyplot as plt
from joblib import dump
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier

logfile = 'model2D_knn_log.txt'
(X_train, X_test, y_train, y_target) = pp.get2Ddata(test_size=0.2, split='time')

# Grid Search
param_grid = [
    {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'weights': ['uniform', 'distance']}
]

# Fit and predict 
clf= GridSearchCV(KNeighborsClassifier(), param_grid, verbose=9, n_jobs=-1)
clf.fit(X_train, y_train)
y_predicted = clf.predict(X_test)

# Create log
pp.makelog(clf,y_target, y_predicted, logfile)

# Save the testing dataset with true target values and predictions
pred = pd.DataFrame(X_test)
pred['Prediction'] = y_predicted
pred['True'] = y_target
pred.to_csv('model2D_knn_predictions.csv')

# Save the trained model for future use
# NOT for this model - saved file too large
dump(clf,'model2D_knn_trained.joblib')