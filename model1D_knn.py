from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import model1D_preprocessing as pp
import matplotlib.pyplot as plt

(X_train, X_test, y_train, y_target) = pp.get1Ddata(test_size=0.5)

ks = []
accs = []
models = []
for k in range(20,0,-1):
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train)
    y_predicted = neigh.predict(X_test)

    acc = accuracy_score(y_target,y_predicted)
    print('k = {}, Accuracy: {}'.format(k, acc))
    ks.append(k)
    accs.append(acc)
    models.append(neigh)

idx = accs.index(max(accs))
print('Best accuracy: {} achieved with k = {}'.format(accs[idx], ks[idx]))

y_predicted = models[idx].predict(X_test)

plt.figure(1)
plt.scatter(y_target,y_predicted,marker='x')
plt.xlabel('Real distance [m]')
plt.ylabel('Predicted distance [m]')
plt.xticks(ticks=[0,50,100,150,200,250,300,350], labels=['0','5','10','15','20','25','30','35'])
plt.yticks(ticks=[0,50,100,150,200,250,300,350], labels=['0','5','10','15','20','25','30','35'])
plt.savefig('Figures/kNN_1D_scatter.pdf',bbox_inches='tight')

plt.figure(2)
plt.plot(ks, accs)
plt.ylim([0.9,1])
plt.xlim([0,20])
plt.xticks(range(2,21,2))
plt.xlabel('Number of neighbors, $k$')
plt.ylabel('Accuracy')
plt.savefig('Figures/kNN_1D_neighbors.pdf',bbox_inches='tight')
