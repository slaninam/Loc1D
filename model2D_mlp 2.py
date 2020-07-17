from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import model2D_preprocessing as pp
import matplotlib.pyplot as plt
from joblib import dump
import pandas as pd

from sklearn.neural_network import MLPClassifier

logfile = 'model2D_mlp_log.txt'
(X_train, X_test, y_train, y_target) = pp.get2Ddata(test_size=0.2, split='time')

# Grid Search
param_grid = [
    {'hidden_layer_sizes': [(1280,1280,1280), (5,10), (10,20), (20,40), (40,80), (80,160), (160,320), (320,640), (640,1280), (1280,1280), (5,10,5), (10,20,10), (40,80,40), (40,80,80), (80,160,160), (60,160,80), (160,320,160)]
    },
    {'activation': ['relu']
    }
]

# Fit and predict 
clf= GridSearchCV(MLPClassifier(max_iter=300), param_grid, verbose=9, n_jobs=-1)
clf.fit(X_train, y_train)
dump(clf,'model2D_mlp_trained.joblib')

y_predicted = clf.predict(X_test)

# Create log
pp.makelog(clf,y_target, y_predicted, logfile)

# Save the testing dataset with true target values and predictions
pred = pd.DataFrame(X_test)
pred['Prediction'] = y_predicted
pred['True'] = y_target
pred.to_csv('model2D_mlp_predictions.csv')

# Save the trained model for future use
#dump(clf,'model2D_svm_trained.joblib')
