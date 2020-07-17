from joblib import load
import matplotlib.pyplot as plt

clf = load('model1D_knn_trained.joblib')
mts = clf.cv_results_['mean_test_score']
neighbors = clf.cv_results_['param_n_neighbors']

mts = mts[0:20:2]
neighbors = neighbors[0:20:2]

ax=plt.subplot(111)
plt.scatter(neighbors,mts,marker='x',c='black')
plt.ylim([0.98, 1])
plt.yticks([0.98, 0.985, 0.99, 0.995, 1.0])
plt.xticks(range(1,11))
#plt.grid(which='major', alpha=0.3)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xlabel('Number of neighbors $k$')
plt.ylabel('Validation accuracy')
#plt.show()
plt.savefig('Figures/kNN_1D_neighbors.pdf',bbox_inches='tight') 