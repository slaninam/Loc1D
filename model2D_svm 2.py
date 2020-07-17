from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import model2D_preprocessing as pp
import matplotlib.pyplot as plt
from joblib import dump
import pandas as pd

from sklearn.svm import SVC

logfile = 'model2D_svm_log.txt'
(X_train, X_test, y_train, y_target) = pp.get2Ddata(test_size=0.2, split='time')

# Grid Search
param_grid = [
    {'C': range(10,40,2),
    'gamma': ['scale'],
    'kernel': ['rbf'],
    'break_ties': [True]}
]

# Fit and predict 
clf= GridSearchCV(SVC(), param_grid, verbose=9, n_jobs=1)
clf.fit(X_train, y_train)
dump(clf,'model2D_svm_trained.joblib')

y_predicted = clf.predict(X_test)

# Create log
pp.makelog(clf,y_target, y_predicted, logfile)

# Save the testing dataset with true target values and predictions
pred = pd.DataFrame(X_test)
pred['Prediction'] = y_predicted
pred['True'] = y_target
pred.to_csv('model2D_svm_predictions.csv')

# Save the trained model for future use
#dump(clf,'model2D_svm_trained.joblib')
