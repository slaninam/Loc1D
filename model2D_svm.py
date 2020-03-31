from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import model2D_preprocessing as pp2
import matplotlib.pyplot as plt
import numpy as np

(X_train, X_test, y_train, y_target) = pp2.get2Ddata(test_size=0.5)

y_train = y_train.astype(int)
y_target = y_target.astype(int)

cs = []
accs = []
model0 = []
model1 = []
for c in range(100,10,-10):
    c=c/10
    clf0 = SVC(C=c, kernel='rbf', gamma='scale', break_ties=True)
    clf1 = SVC(C=c, kernel='rbf', gamma='scale', break_ties=True)

    clf0.fit(X_train, y_train[:,0])
    clf1.fit(X_train, y_train[:,1])
    #y_predicted = [clf0.predict(X_test), clf1.predict(X_test)]
    #print(y_predicted)
    p1 = clf0.predict(X_test)
    p2 = clf1.predict(X_test)
    #y_predicted = np.concatenate((clf0.predict(X_test).T, clf1.predict(X_test).T))
 
    y_predicted = np.concatenate((p1.reshape(-1,1), p2.reshape(-1,1)), axis=1)

    accx = accuracy_score(y_target[:,0],y_predicted[:,0])
    accy = accuracy_score(y_target[:,1],y_predicted[:,1])
    accc = accuracy_score(100*y_target[:,0]+y_target[:,1], 100*y_predicted[:,0]+y_predicted[:,1])
    print('c = {}, Accuracy x: {}'.format(c, accx))
    print('c = {}, Accuracy y: {}'.format(c, accy))
    print('c  = {}, Compound accuracy: {}'.format(c, accc))
    cs.append(c)
    accs.append(accc)
    model0.append(clf0)
    model1.append(clf1)

idx = accs.index(max(accs))
print('Best accuracy: {} achieved with c = {}'.format(accs[idx], cs[idx]))

#y_predicted = models[idx].predict(X_test)
#y_predicted = [model0[idx].predict(X_test), model1[idx].predict(X_test)]
y_predicted = np.concatenate((clf0.predict(X_test).reshape(-1,1), clf1.predict(X_test).reshape(-1,1)), axis=1)

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
plt.savefig('Figures/svm_2D_scatter.pdf',bbox_inches='tight')
plt.show()

#plt.figure(2)
#plt.plot(ks, accs)
#plt.ylim([0.9,1])
#plt.xlim([0,20])
#plt.xticks(range(2,21,2))
#plt.xlabel('Number of neighbors, $k$')
#plt.ylabel('Accuracy')
#plt.savefig('Figures/kNN_1D_neighbors.pdf',bbox_inches='tight')
