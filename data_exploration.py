import pandas as pd
import numpy as np

folder = 'SE7.106'

# Initialize based on folder to process:
if folder == 'data_new':
    dists_x = np.arange(0.0, 35.5, 0.5)
    dists_y = (0.4, 0.9, 1.4)
    prefix = 'corridor'
    dist_y_offset = 0
elif folder == 'SE7.105':
    dists_x = np.arange(1.0, 9.0, 0.5)
    dists_y = (3.1, 3.6, 4.4, 4.9, 5.4, 5.9, 6.4, 6.9)
    prefix = 'lab7105'
    dist_y_offset = 0
elif folder == 'SE7.106':
    dists_x = (10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0)
    dists_y = (6.4, 7.4, 8.4, 9.4, 10.4)
    prefix = 'lab7106'
    dist_y_offset = -3.3

# Read data from measurement files
dfs = []
for dist_x in dists_x:
    for dist_y in dists_y:
        df = pd.DataFrame()
        if dist_x.is_integer():
            fname = folder+'/dist.' + '{0:.0f}'.format(dist_x) + 'mX{0:.1f}m.txt'.format(dist_y)
        else:
            fname = folder+'/dist.' + '{0:.1f}'.format(dist_x) + 'mX{0:.1f}m.txt'.format(dist_y) # Added suffix to take center of corridor onl
        try: # take care of exceptions since not all file names are present
            df = pd.read_csv(fname, header = None, names=['time','tag id','anchor id','anchor1 id', 'anchor2 id', 'anchor3 id','a0ch0','a0ch1','a0ch2','a0ch3','a0rssi0','a0rssi1','a0rssi2','a0rssi3','a1ch0','a1ch1','a1ch2','a1ch3','a1rssi0','a1rssi1','a1rssi2','a1rssi3','a2ch0','a2ch1','a2ch2','a2ch3','a2rssi0','a2rssi1','a2rssi2','a2rssi3','a3ch0','a3ch1','a3ch2','a3ch3','a3rssi0','a3rssi1','a3rssi2','a3rssi3'])
        except:
            print(fname+' not found.')
        if len(df)>0: # only add if the data is actually read
            df['dist_x'] = dist_x
            df['dist_y'] = format(dist_y + dist_y_offset, '.1f')
            dfs.append(df)       
df = pd.concat(dfs).reset_index(drop=True)
print('{} items read.'.format(len(df)))

# Clean data in a faster way than in the first implementation.
# Example for one record:
# df.loc[df['a0ch0']==0, 'a0c0'] = df['a0rssi0']

for anchor in range(0,4):
    for meas in range(0,4):
        for chan in range(0,3):
            meas_a = 'a' + str(anchor) + 'ch' + str(meas)
            meas_rss = 'a' + str(anchor) + 'rssi' + str(meas)
            target = 'a' + str(anchor) + 'c' + str(chan)
            df.loc[df[meas_a]==chan, target] = df[meas_rss]

# drop original data
print(df['anchor id'].unique(), df['anchor1 id'].unique(), df['anchor2 id'].unique(), df['anchor3 id'].unique())
df.drop(['tag id', 'anchor id', 'anchor1 id', 'anchor2 id',
       'anchor3 id', 'a0ch0', 'a0ch1', 'a0ch2', 'a0ch3', 'a0rssi0', 'a0rssi1',
       'a0rssi2', 'a0rssi3', 'a1ch0', 'a1ch1', 'a1ch2', 'a1ch3', 'a1rssi0',
       'a1rssi1', 'a1rssi2', 'a1rssi3', 'a2ch0', 'a2ch1', 'a2ch2', 'a2ch3',
       'a2rssi0', 'a2rssi1', 'a2rssi2', 'a2rssi3', 'a3ch0', 'a3ch1', 'a3ch2',
       'a3ch3', 'a3rssi0', 'a3rssi1', 'a3rssi2', 'a3rssi3'], axis=1, inplace = True)

# reindex column so that the anchors are properly indexed (update March 10)
for channel in range(0,3):
    chan = str(channel)
    df['a4c'+chan] = df['a3c'+chan]
    df['a3c'+chan] = df['a2c'+chan]
    df['a2c'+chan] = df['a1c'+chan]
    df['a1c'+chan] = df['a0c'+chan]
    df['a0c'+chan] = df['a4c'+chan]
    df.drop(['a4c'+chan], axis=1, inplace=True)


# sort by alphabet
df = df[sorted(df.columns.tolist())]

df.to_csv(prefix+'_rough.csv',sep=';')
print('{} items processed.'.format(len(df)))

# Clean data
df.replace([-110], [None], inplace=True)

df['idx'] = range(0,len(df))
df['idx'] = df['idx']//2
df2 = df.groupby(['dist_x','dist_y','idx'], as_index=False).fillna(method='backfill')

df2 = df2.iloc[0:len(df2):2,:]

print('Complete before: {}'.format(len(df.dropna())))
print('Complete after : {}'.format(len(df2.dropna())))

df2.dropna().drop(['idx'],axis=1).to_csv(prefix+'_clean.csv',sep=';')