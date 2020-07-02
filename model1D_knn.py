from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
import model1D_preprocessing as pp
import matplotlib.pyplot as plt
from joblib import dump

logfile = 'model1D_knn_log.txt'
(X_train, X_test, y_train, y_target) = pp.get1Ddata(test_size=0.2, split='time')

# Grid Search
param_grid = [
    {'n_neighbors': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
    'weights': ['uniform', 'distance']}
]
clf = GridSearchCV(KNeighborsClassifier(), param_grid, verbose=9, n_jobs=-1)
clf.fit(X_train, y_train)

y_predicted = clf.predict(X_test)

plt.figure(1)
plt.scatter(y_target,y_predicted,marker='x')
plt.xlabel('Real distance [m]')
plt.ylabel('Predicted distance [m]')
plt.xticks(ticks=[0,50,100,150,200,250,300,350], labels=['0','5','10','15','20','25','30','35'])
plt.yticks(ticks=[0,50,100,150,200,250,300,350], labels=['0','5','10','15','20','25','30','35'])
plt.savefig('Figures/kNN_1D_scatter.pdf',bbox_inches='tight')

print('Estimator details:', file=open(logfile, 'a'))
print(clf, file=open(logfile, 'a'))
print('Test set accuracy: {}'.format(accuracy_score(y_target, y_predicted)), file=open(logfile, 'a'))
print('Best parameters based on cross-validation:', file=open(logfile, 'a'))
print(clf.best_params_, file=open(logfile, 'a'))
print('*************',file=open(logfile, 'a'))
print('CV results:',file=open(logfile, 'a'))
print('*************',file=open(logfile, 'a'))
print(clf.cv_results_, file=open(logfile, 'a'))

# Save the testing dataset with true target values and predictions
pred = X_test
pred['Prediction'] = y_predicted
pred['True'] = y_target
pred.to_csv('model1D_knn_predictions.csv')

# Save the trained model for future use
dump(clf,'model1D_knn_trained.joblib')

