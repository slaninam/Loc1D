from sklearn.metrics import accuracy_score
import model1D_preprocessing as pp
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from joblib import dump, load

# Define paths, load data
logfile = 'model1D_mlp_log.txt'
(X_train, X_test, y_train, y_target) = pp.get1Ddata(test_size=0.2)

# GridSearch
param_grid = [
  {'hidden_layer_sizes': [(5,10), (10,20), (20,40), (40,80), (80,160), (160,320), 
       (320,640), (640,1280), (1280,1280), (5,10,5), (10,20,10), (40,80,40), (40,80,80), (80,160,160), (80,160,80), (160,320,160), (1280,1280,1280)], 'activation': ['relu']}
 ]
clf = GridSearchCV(
        MLPClassifier(max_iter=300), param_grid, verbose=9, n_jobs=-1
    )
clf.fit(X_train, y_train)


y_predicted = clf.predict(X_test)
print('Estimator details:', file=open(logfile, 'a'))
print(clf, file=open(logfile, 'a'))
print('Test set accuracy: {}'.format(accuracy_score(y_target, y_predicted)), file=open(logfile, 'a'))
print('Best parameters based on cross-validation:', file=open(logfile, 'a'))
print(clf.best_params_, file=open(logfile, 'a'))



plt.figure(1)
plt.scatter(y_target,y_predicted,marker='x')
plt.xlabel('Real distance [m]')
plt.ylabel('Predicted distance [m]')
plt.xticks(ticks=[0,50,100,150,200,250,300,350], labels=['0','5','10','15','20','25','30','35'])
plt.yticks(ticks=[0,50,100,150,200,250,300,350], labels=['0','5','10','15','20','25','30','35'])
#plt.show()
plt.savefig('Figures/mlp_1D_scatter.pdf',bbox_inches='tight')

# Save the testing dataset with true target values and predictions
pred = X_test
pred['Prediction'] = y_predicted
pred['True'] = y_target
pred.to_csv('model1D_mlp_predictions.csv')

# Save the trained model for future use
dump(clf,'model1D_mlp_trained.joblib')