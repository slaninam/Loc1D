from matplotlib import cm
from cycler import cycler
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

anchor = '3'

df = pd.read_csv('corridor_clean.csv',sep=';')

dispdata = df.groupby(['dist_x','dist_y'], as_index=False).mean()
dispdata = dispdata[dispdata['dist_y']==0.9].set_index('dist_x').loc[:,('a'+
anchor+'c0','a'+anchor+'c1','a'+anchor+'c2')]

colors = plt.cm.cubehelix(np.linspace(0,1,4))
plt.figure(figsize=(10,4))

#plt.gca().set_prop_cycle('color', colors)
#_=plt.plot(dispdata)

xval = dispdata.reset_index().loc[:,'dist_x']
plt.step(xval,dispdata.loc[:,'a'+anchor+'c0'],color=colors[0],label='Channel 0')
plt.plot(xval,dispdata.loc[:,'a'+anchor+'c0'],'o--',color=colors[0],alpha=0.3)
plt.step(xval,dispdata.loc[:,'a'+anchor+'c1'],color=colors[1],label='Channel 1')
plt.plot(xval,dispdata.loc[:,'a'+anchor+'c1'],'o--',color=colors[1],alpha=0.3)
plt.step(xval,dispdata.loc[:,'a'+anchor+'c2'],color=colors[2],label='Channel 2')
plt.plot(xval,dispdata.loc[:,'a'+anchor+'c2'],'o--',color=colors[2],alpha=0.3)
plt.xlabel('Distance coordinate [m], anchor '+anchor+'.')
plt.ylabel('RSS [dBm]')
plt.legend(loc='upper right')
plt.savefig('rss_levels_1D_anchor'+anchor+'.pdf', bbox_inches='tight')
