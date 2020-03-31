from sklearn.metrics import accuracy_score
import model1D_preprocessing as pp
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier

(X_train, X_test, y_train, y_target) = pp.get1Ddata(test_size=0.5)

ns = []
accs = []
models = []

# may try to vary gamma (kernel coefficient), C (regulatization parameter)
for n in range(200,0,-10):
    clf = RandomForestClassifier(n_estimators=n)
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)

    acc = accuracy_score(y_target,y_predicted)
    print('n = {}, Accuracy: {}'.format(n, acc))
    ns.append(n)
    accs.append(acc)
    models.append(clf)

idx = accs.index(max(accs))
print('Best accuracy: {} achieved with n = {}'.format(accs[idx], ns[idx]))

y_predicted = models[idx].predict(X_test)

plt.figure(1)
plt.scatter(y_target,y_predicted,marker='x')
plt.xlabel('Real distance [m]')
plt.ylabel('Predicted distance [m]')
plt.xticks(ticks=[0,50,100,150,200,250,300,350], labels=['0','5','10','15','20','25','30','35'])
plt.yticks(ticks=[0,50,100,150,200,250,300,350], labels=['0','5','10','15','20','25','30','35'])
#plt.show()
plt.savefig('Figures/rf_1D_scatter.pdf',bbox_inches='tight')

plt.figure(2)
plt.plot(ns, accs)
plt.ylim([0.8,1])
#plt.xlim([0,2])
plt.xlabel('Number of estimators, $n$')
plt.ylabel('Accuracy')
plt.savefig('Figures/rf_1D_estimators.pdf',bbox_inches='tight')
