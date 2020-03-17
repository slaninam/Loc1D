import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns

for anc in range(0,4):
    anchor = str(anc)
    lab = '7105'

    df = pd.read_csv('lab'+lab+'_clean.csv',sep=';')

    dispdata = df.groupby(['dist_x','dist_y'], as_index=False).mean()
    dispdata = dispdata.set_index(['dist_x', 'dist_y']).loc[:,'a'+anchor+'c0']
    dispdata = dispdata.unstack().transpose()

    print(dispdata)

    plt.figure()
    f = sns.heatmap(dispdata, vmin=-90, vmax=-52)
    plt.xlabel('Distance $x$ [m]')
    plt.ylabel('Distance $y$ [m]')
    for n, label in enumerate(f.xaxis.get_ticklabels()):
        if n % 2 !=0: 
            label.set_visible(False)
    plt.savefig('rss_levels_2D_'+lab+'_anchor'+anchor+'.pdf', bbox_inches='tight')