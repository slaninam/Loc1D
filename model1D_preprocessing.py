import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# read data, pick a single y coordinate, drop the remaining y coordinates
def get1Ddata(test_size=0.2, split='time'):
    df = pd.read_csv('data_csv/corridor_clean.csv', sep=';').drop(['Unnamed: 0'],axis= 'columns')
    df = df[df['dist_y']==0.9]
    df = df.drop(['dist_y'],axis=1)
    print('Samples total: {}'.format(len(df)))
    print('No.categories: {}'.format(len(df['dist_x'].unique())))

    if split=='rand':
        X_train, X_test, y_train, y_test = train_test_split(df.drop(axis=1,labels = ['dist_x']), df['dist_x']*10, test_size=test_size)
        X_train.drop(['time'],axis= 'columns',inplace=True)
        X_test.drop(['time'],axis= 'columns',inplace=True)
         # print the training / test sample sizes
        print('Samples train: {}'.format(len(y_train)))
        print('Samples test:  {}'.format(len(y_test)))

    # split based on time stamps
    if split=='time':
        train = []
        test = []
        X_train = pd.DataFrame()
        for pos in df['dist_x'].unique():
            # assuming input data ordered by time stamp (true based on exploration)
            chunk = df[df['dist_x']==pos]
            testitems = int(test_size*len(chunk))
            trainitems = len(chunk)-testitems
            # Store chunk splits
            train.append(chunk.iloc[:trainitems])
            test.append(chunk.iloc[-testitems:])
        # Merge components together           
        train = pd.concat(train).reset_index(drop=True)
        test  = pd.concat(test).reset_index(drop=True)
        # Create train and test sets
        X_train = train.drop(axis=1, labels=['dist_x', 'time'])
        y_train = train['dist_x']*10
        X_test  = test.drop(axis=1, labels=['dist_x', 'time'])
        y_test  = test['dist_x']*10
        print('Samples train: {}'.format(len(train)))
        print('Samples test:  {}'.format(len(test)))

    # convert the targets to ordered categorical variables
    y_train = pd.Categorical(y_train,ordered=True)
    y_target = pd.Categorical(y_test,ordered=True)

    return (X_train, X_test, y_train, y_target)

# define a simple plotting routine
def showplot(x,y):
    plt.scatter(x,y,marker='x')
    plt.xlabel('Real distance [m]')
    plt.ylabel('Predicted distance [m]')
    plt.show()

def showdep(x,y,xlabel='',ylabel=''):
    plt.plot(x,y,marker='x')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

if __name__ == "__main__":
    get1Ddata()