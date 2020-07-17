import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd

clf='knn'

# Decode the x and y values from a sincle class ID
df = pd.read_csv('model2D_'+clf+'_predictions.csv')
df['Prediction_y'] = (df['Prediction'] % 1000) /10
df['Prediction_x'] = (df['Prediction'] - 10*df['Prediction_y'])/10000

df['True_y'] = (df['True'] % 1000) /10
df['True_x'] = (df['True'] - 10*df['True_y'])/10000
print(df)

# Count the occurence of points in the scatter 
dfx = df[['Prediction_x', 'True_x']]
dfx['idstr'] = dfx['Prediction_x']+1000*dfx['True_x']
dfy = df[['Prediction_y', 'True_y']]
dfy['idstr'] = dfy['Prediction_y']+1000*dfy['True_y']


countsx = dfx.groupby('idstr').size().reset_index()
countsy = dfy.groupby('idstr').size().reset_index()

# #counts = pd.concat([counts, counts.index()], axis=1)
countsx['Predicted_x']      = countsx['idstr']%100 
countsx['True_x']           = (countsx['idstr'] - countsx['Predicted_x']) / 1000
countsx['Predicted_x'] = countsx['Predicted_x']
countsx['True_x'] = countsx['True_x']

countsy['Predicted_y']      = countsy['idstr']%100 
countsy['True_y']           = (countsy['idstr'] - countsy['Predicted_y']) / 1000
countsy['Predicted_y'] = countsy['Predicted_y']
countsy['True_y'] = countsy['True_y']

# Normalize counts
countsx[0] = countsx[0] / countsx[0].max()
countsy[0] = countsy[0] / countsy[0].max()
print(countsx)



# # Display scatter
plt.figure(1)
#plt.subplots(1,2,figsize=(8,5))
ax1=plt.subplot(121)
sc=plt.scatter(countsx['True_x'],countsx['Predicted_x'],c=countsx[0],cmap='rainbow', marker='.')
plt.xlabel('Real distance $x$ [m]')
plt.ylabel('Predicted distance $x$ [m]')
plt.grid(which='major', alpha=0.3)
ax2=plt.subplot(122)
sc=plt.scatter(countsy['True_y'],countsy['Predicted_y'],c=countsy[0],cmap='rainbow', marker='.')
#sc=plt.scatter(df['True_y'],df['Prediction_y'],marker='.')
plt.colorbar(sc)
plt.xlabel('Real distance $y$ [m]')
plt.ylabel('Predicted distance $y$ [m]')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)
plt.grid(which='major', alpha=0.3)
#plt.show()
plt.savefig('Figures/'+clf+'_2D_scatter.pdf',bbox_inches='tight') 