import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# read data, pick a single y coordinate, drop the remaining y coordinates
def get1Ddata(test_size=0.3):
    df = pd.read_csv('data_csv/corridor_clean.csv', sep=';').drop(['Unnamed: 0','time'],axis= 'columns')
    df = df[df['dist_y']==0.9]
    df = df.drop(['dist_y'],axis=1)

    # print out an overview of the total sample sizes and the training / test sample sizes
    print('Samples total: {}'.format(len(df)))
    X_train, X_test, y_train, y_test = train_test_split(df.drop(axis=1,labels = ['dist_x']), df['dist_x']*10, test_size=test_size)
    print('Samples train: {}'.format(len(y_train)))
    print('Samples test:  {}'.format(len(y_test)))

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