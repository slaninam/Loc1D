import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def get2Ddata(test_size=0.2, split='time'):
    df1 = pd.read_csv('data_csv/lab7105_clean.csv', sep=';').drop(['Unnamed: 0'],axis= 'columns')
    df2 = pd.read_csv('data_csv/lab7106_clean.csv', sep=';').drop(['Unnamed: 0'],axis= 'columns')
    df3 = pd.read_csv('data_csv/corridor_clean.csv', sep=';').drop(['Unnamed: 0'],axis= 'columns')
    df = pd.concat([df1, df2, df3])
    print(df.tail())

    if split=='rand':
        # print out an overview of the total sample sizes and the training / test sample sizes
        print('Samples total: {}'.format(len(df)))
        X_train, X_test, y_train, y_test = train_test_split(df.drop(axis=1,labels = ['dist_x','dist_y']), df.loc[:,('dist_x','dist_y')]*10, test_size=test_size)
        X_train.drop(['time'],axis= 'columns',inplace=True)
        X_test.drop(['time'],axis= 'columns',inplace=True)
        print('Samples train: {}'.format(len(y_train)))
        print('Samples test:  {}'.format(len(y_test)))

    if split=='time':
        train = []
        test = []
        X_train = pd.DataFrame()
        for posx in df['dist_x'].unique():
            for posy in df['dist_y'].unique():
                # Each chunk processed at once corresponds to one position
                chunk = df[(df['dist_x']==posx) & (df['dist_y']==posy)]
                # Skip chunk if empty
                if len(chunk) < 1:
                    continue 
                # Split and store
                testitems = int(test_size*len(chunk))
                trainitems= len(chunk)-testitems
                train.append(chunk.iloc[:trainitems])
                test.append(chunk.iloc[-testitems:])
        # Merge components together
        train = pd.concat(train).reset_index(drop=True)
        test  = pd.concat(test).reset_index(drop=True)
        # Create train and test sets
        X_train = train.drop(axis=1, labels=['dist_x', 'dist_y', 'time'])
        y_train = train['dist_x']*10000+train['dist_y']*10
        X_test  = test.drop(axis=1, labels=['dist_x', 'dist_y', 'time'])
        y_test  = test['dist_x']*10000+test['dist_y']*10
        print('Samples train: {}'.format(len(train)))
        print('Samples test:  {}'.format(len(test)))

    
    # convert the targets to ordered categorical variables
    y_train = pd.Categorical(y_train,ordered=False)
    y_target = pd.Categorical(y_test,ordered=False)


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

def makelog(clf, y_target, y_predicted, logfile='log.txt'):
    print('Estimator details:', file=open(logfile, 'a'))
    print(clf, file=open(logfile, 'a'))
    print('Test set accuracy: {}'.format(accuracy_score(y_target, y_predicted)), file=open(logfile, 'a'))
    print('Best parameters based on cross-validation:', file=open(logfile, 'a'))
    print(clf.best_params_, file=open(logfile, 'a'))
    print('*************',file=open(logfile, 'a'))
    print('CV results:',file=open(logfile, 'a'))
    print('*************',file=open(logfile, 'a'))
    print(clf.cv_results_, file=open(logfile, 'a'))

if __name__ == "__main__":
    (X_train, X_test, y_train, y_target) = get2Ddata()
    y_all = np.concatenate((y_train, y_target))
    y_all = pd.Series(y_all)
    classes = len(y_all.unique())
    print('Output classes: {}'.format(classes))