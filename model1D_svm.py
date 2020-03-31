from sklearn.metrics import accuracy_score
import model1D_preprocessing as pp
import matplotlib.pyplot as plt

from sklearn.svm import SVC

(X_train, X_test, y_train, y_target) = pp.get1Ddata(test_size=0.5)

cs = []
accs = []
models = []

# may try to vary gamma (kernel coefficient), C (regulatization parameter)
for c in range(20,0,-2):
    c=c/10
    clf = SVC(C=c, kernel='rbf', gamma='scale', break_ties=True)
    clf.fit(X_train, y_train)
    y_predicted = clf.predict(X_test)

    acc = accuracy_score(y_target,y_predicted)
    print('c = {}, Accuracy: {}'.format(c, acc))
    cs.append(c)
    accs.append(acc)
    models.append(clf)

idx = accs.index(max(accs))
print('Best accuracy: {} achieved with c = {}'.format(accs[idx], cs[idx]))

y_predicted = models[idx].predict(X_test)

plt.figure(1)
plt.scatter(y_target,y_predicted,marker='x')
plt.xlabel('Real distance [m]')
plt.ylabel('Predicted distance [m]')
plt.xticks(ticks=[0,50,100,150,200,250,300,350], labels=['0','5','10','15','20','25','30','35'])
plt.yticks(ticks=[0,50,100,150,200,250,300,350], labels=['0','5','10','15','20','25','30','35'])
#plt.show()
plt.savefig('Figures/svm_1D_scatter.pdf',bbox_inches='tight')

plt.figure(2)
plt.plot(cs, accs)
plt.ylim([0.8,1])
plt.xlim([0,2])
plt.xlabel('Regulatization parameter, $c$')
plt.ylabel('Accuracy')
plt.savefig('Figures/svm_1D_regulatization.pdf',bbox_inches='tight')
