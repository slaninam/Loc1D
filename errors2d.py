import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd
import numpy as np

clf='mlp'

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


df['Distance_x'] = abs(df['Prediction_x'] - df['True_x']) 
df['Distance_y'] = abs(df['Prediction_y'] - df['True_y']) 
df['Distance'] = np.sqrt(df['Distance_x']**2 + df['Distance_y']**2)

print(df[df['Distance']>0])

print('Max. error [m]: {}'.format(df['Distance'].max()))
print('Mean error [m]: {}'.format(df['Distance'].mean()))

