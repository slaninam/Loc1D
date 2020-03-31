from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import model2D_preprocessing as pp2
import matplotlib.pyplot as plt

(X_train, X_test, y_train, y_target) = pp2.get2Ddata(test_size=0.5)

y_train = y_train.astype(int)
y_target = y_target.astype(int)

ks = []
accs = []
models = []
for k in range(5,0,-1): #ange(3,5):#
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(X_train, y_train)
    y_predicted = neigh.predict(X_test)

    accx = accuracy_score(y_target[:,0],y_predicted[:,0])
    accy = accuracy_score(y_target[:,1],y_predicted[:,1])
    accc = accuracy_score(100*y_target[:,0]+y_target[:,1], 100*y_predicted[:,0]+y_predicted[:,1])
    print('k = {}, Accuracy x: {}'.format(k, accx))
    print('k = {}, Accuracy y: {}'.format(k, accy))
    print('k = {}, Compound accuracy: {}'.format(k, accc))
    ks.append(k)
    accs.append(accc)
    models.append(neigh)

idx = accs.index(max(accs))
print('Best accuracy: {} achieved with k = {}'.format(accs[idx], ks[idx]))

y_predicted = models[idx].predict(X_test)

fig, (ax1,ax2) = plt.subplots(1,2)
ax1.scatter(y_target[:,0],y_predicted[:,0],marker='x', alpha=.1)
ax1.set_xlabel('Real distance $x$  [m]')
ax1.set_ylabel('Predicted distance $x$ [m]')
ax1.set_xticks(ticks=[0,50,100,150,200,250,300,350])
ax1.set_xticklabels(labels=['0','5','10','15','20','25','30','35'])
ax1.set_yticks(ticks=[0,50,100,150,200,250,300,350])
ax1.set_yticklabels(labels=['0','5','10','15','20','25','30','35'])

ax2.scatter(y_target[:,1],y_predicted[:,1],marker='x', alpha=.1)
ax2.set_xlabel('Real distance $y$  [m]')
ax2.set_ylabel('Predicted distance $y$ [m]')
ax2.set_xticks(ticks=[0,20,40,60,80])
ax2.set_xticklabels(labels=['0','2','4','6','8'])
ax2.set_yticks(ticks=[0,20,40,60,80])
ax2.set_yticklabels(labels=['0','2','4','6','8'])
plt.savefig('Figures/kNN_2D_scatter.pdf',bbox_inches='tight')
plt.show()

#plt.figure(2)
#plt.plot(ks, accs)
#plt.ylim([0.9,1])
#plt.xlim([0,20])
#plt.xticks(range(2,21,2))
#plt.xlabel('Number of neighbors, $k$')
#plt.ylabel('Accuracy')
#plt.savefig('Figures/kNN_1D_neighbors.pdf',bbox_inches='tight')
