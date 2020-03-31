import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# read data, pick a single y coordinate, drop the remaining y coordinates
def get2Ddata(test_size=0.3):
    df1 = pd.read_csv('data_csv/lab7105_clean.csv', sep=';').drop(['Unnamed: 0','time'],axis= 'columns')
    df2 = pd.read_csv('data_csv/lab7106_clean.csv', sep=';').drop(['Unnamed: 0','time'],axis= 'columns')
    df3 = pd.read_csv('data_csv/corridor_clean.csv', sep=';').drop(['Unnamed: 0','time'],axis= 'columns')
    
    df = pd.concat([df1, df2, df3])
    print(df.tail())

    # print out an overview of the total sample sizes and the training / test sample sizes
    print('Samples total: {}'.format(len(df)))
    X_train, X_test, y_train, y_test = train_test_split(df.drop(axis=1,labels = ['dist_x','dist_y']), df.loc[:,('dist_x','dist_y')]*10, test_size=test_size)
    print('Samples train: {}'.format(len(y_train)))
    print('Samples test:  {}'.format(len(y_test)))

    # convert the targets to ordered categorical variables
    #y_train['dist_x'] = pd.Categorical(y_train['dist_x'],ordered=False)
    #y_train['dist_y'] = pd.Categorical(y_train['dist_y'],ordered=False)
    #y_target = pd.Categorical(y_test,ordered=False)

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_target = y_test.to_numpy()

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
    get2Ddata()