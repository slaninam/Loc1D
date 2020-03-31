import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd


df = pd.read_csv('corridor_clean.csv',sep=';')

dispdata = df.iloc[::5, :] # 5 times sumbsampling
dispdata = dispdata[dispdata['dist_y']==0.9]

fig=plt.figure()
ax=Axes3D(fig)
print(dispdata.columns)
p = ax.scatter(dispdata.loc[:,'a0c0'], dispdata.loc[:,'a1c0'], dispdata.loc[:,'a3c0'], c=dispdata.loc[:,'dist_x']/35)
ax.set_xlabel('Anchor 0 RSS [dBm]')
ax.set_ylabel('Anchor 1 RSS [dBm]')
ax.set_zlabel('Anchor 3 RSS [dBm]')
ax=plt.gca()
#divider = make_axes_locatable(ax)
#cax = divider.append_axes("right", size="5%", pad=0.05)
cbar = fig.colorbar(p, shrink=0.7)
cbar.ax.set_yticklabels(['0 m', '7 m', '14 m', '21 m', '28 m', '35 m'])
plt.show()
