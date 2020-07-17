from joblib import load
import matplotlib.pyplot as plt

clf = load('model1D_svm_trained.joblib')
mts = clf.cv_results_['mean_test_score']
#print(clf.cv_results_)
paramc = clf.cv_results_['param_C']

mts = mts[0:46:3]
paramc = paramc[0:46:3]

ax=plt.subplot(111)
plt.scatter(paramc,mts,marker='x',c='black')
plt.ylim([0.98, 1])
plt.yticks([0.98, 0.985, 0.99, 0.995, 1.0])
# plt.xticks(range(1,11))
# #plt.grid(which='major', alpha=0.3)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.xlabel('Regularization parameter $C$')
plt.ylabel('Validation accuracy')
#plt.show()
plt.savefig('Figures/svm_1D_regularization.pdf',bbox_inches='tight') 