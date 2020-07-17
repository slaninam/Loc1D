import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd

clf='svm'

df = pd.read_csv('model1D_'+clf+'_predictions.csv')
df = df[['Prediction', 'True']]
df['idstr'] = df['Prediction']+1000*df['True']

counts = df.groupby('idstr').size().reset_index()

#counts = pd.concat([counts, counts.index()], axis=1)
counts['Predicted']      = counts['idstr']%1000 
counts['True']           = (counts['idstr'] - counts['Predicted']) / 1000
counts['Predicted'] = counts['Predicted']/10
counts['True'] = counts['True']/10
# Normalize counts
#counts[0] = counts[0] / counts[0].max()
print(counts)

# Display scatter
plt.figure(1)
ax=plt.subplot(111)
sc=plt.scatter(counts['True'],counts['Predicted'],c=counts[0],cmap='rainbow', marker='.')
plt.colorbar(sc)
plt.xlabel('Real distance [m]')
plt.ylabel('Predicted distance [m]')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.grid(which='major', alpha=0.3)
#plt.show()
plt.savefig('Figures/'+clf+'_1D_scatter.pdf',bbox_inches='tight') 