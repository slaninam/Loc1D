import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd

clf='rf'

df = pd.read_csv('model1D_'+clf+'_predictions.csv')

df['Distance'] = abs(df['Prediction'] - df['True'])/10 
print(df)

print('Max. error [m]: {}'.format(df['Distance'].max()))
print('Mean error [m]: {}'.format(df['Distance'].mean()))
