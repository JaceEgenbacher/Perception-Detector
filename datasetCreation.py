import pandas as pd
import numpy as np
import time
#can be used to crossvalidation
from sklearn.utils import shuffle
from sklearn.cluster import KMeans

#I created a class
class datasetCreation :

    def __init__(self,skip_int = 0,reload = False) :
        ''' here I initialize by just loading the csv between 6s and 9s (6s = where the face shows)'''
        if reload == True :
            tBeforeL = time.time()
            self.rawBlinkLogicalL = pd.read_csv("rawBlinkLogicalL.csv",header = None, sep=",", skiprows = skip_int,usecols = [*range(6000,9000)])
            self.microsaccadeBinoc = pd.read_csv("microsaccadeBinoc.csv",header = None, sep=",", skiprows = skip_int,usecols = [*range(6000,9000)])
            self.interpPupilL = pd.read_csv("interpPupilL.csv", sep=",",header = None, skiprows = skip_int,usecols = [*range(6000,9000)])
            self.condition = pd.read_csv("condition.csv", sep=",",header = None, skiprows = skip_int, names = ['res'])
            tAfterL = time.time()
            # print('finished loading in : ', tAfterL - tBeforeL)

    def replaceBlink(self,value):
        if value == 0 :
            return 0
        else :
            return 1



    def datasetCreation(self,splitNumber):
        ''' key function : we take the mean of blink,microssacade and interPupilL,
        use K-means on them to find what influences the most CNP or CP and classify according to the value'''
        tBeforeC = time.time()
        #replace CP by 1 and CNP by 0
        self.condition[self.condition['res']=='CP'] = 1
        self.condition[self.condition['res']=='CNP'] = 0
        #those 3 csv are just sumed up in 1 value which is the average
        self.rawBlinkLogicalL = self.rawBlinkLogicalL.groupby(np.arange(len(self.rawBlinkLogicalL.columns))//splitNumber, axis=1).mean()
        self.microsaccadeBinoc = self.microsaccadeBinoc.groupby(np.arange(len(self.microsaccadeBinoc.columns))//splitNumber, axis=1).mean()
        self.interpPupilL = self.interpPupilL.groupby(np.arange(len(self.interpPupilL.columns))//splitNumber, axis=1).mean()
        #merge the 3 values and the result in df
        df = pd.concat([self.rawBlinkLogicalL,self.microsaccadeBinoc,self.interpPupilL,self.condition],axis=1,join = 'inner')
        #rename the columns beacause it is easier to manipulate
        # df.columns = ['blink','microssacade','interPupil','res']

        #I use a first classification before K-means bc when you don't blink you have way less chances to be CP
        # df['blink'] = df['blink'].apply(self.replaceBlink)
        #we use kmeans here on every columnn one by one, the number of clusters is chosen by me (I look what seems the best)
        # Xb,Xm,Xi = df[['blink','res']],df[['microssacade','res']],df[['interPupil','res']]
        # kmeansb,kmeansm,kmeansi = KMeans(n_clusters=3, random_state=0).fit(Xb),KMeans(n_clusters=4, random_state=0).fit(Xm),KMeans(n_clusters=4, random_state=0).fit(Xi)
        # df['blink'] = kmeansb.predict(Xb)
        # df['microssacade'] = kmeansm.predict(Xm)
        # df['interPupil'] = kmeansi.predict(Xi)
        # #print the centers to analyze bad points later
        # print('kmeansB',kmeansb.cluster_centers_)
        # print('kmeansM',kmeansm.cluster_centers_)
        # print('kmeansI',kmeansi.cluster_centers_)
        tAfterC = time.time()
        # print('finished cleaning in : ', tAfterC - tBeforeC)
        return df

    def saveDataframe(self,df) :
        t1 = time.time()
        df.to_csv('dfSaved.csv')
        t2 = time.time()
        print('Saved done in : ', t2 - t1)

    def loadDataframe(self):
        t1 = time.time()
        df = pd.read_csv("dfSaved.csv", sep=",",index_col = 0)
        t2 = time.time()
        print('Loading done in : ', t2 - t1)
        return df

    def produceSets(self,df):
        '''the dataset has way more CP than CNP so we want to train/test on the same amount of CP than CNP'''
        dfCP = df[df['res'] == 1]
        dfCNP = df[df['res'] == 0]
        # dfCP = shuffle(dfCP)
        # dfCNP = shuffle(dfCNP)
        #less CNP so 90% used 10% testing
        #issue : we want to train on same number of CP than CNP
        split = round((dfCNP.shape[0])*.9)

        dfShuffled = shuffle(df)
        Xtrain = dfShuffled[split:]
        Xtest = dfShuffled[:split]

        trainCP,trainCNP = dfCP.iloc[:split],dfCNP.iloc[:split]
        Xtrain = pd.concat([trainCP,trainCNP])

        Ytrain = Xtrain['res']
        Xtrain = Xtrain.drop('res',axis =1)

        # testCNP = dfCNP.iloc[split:]
        # we want as much CP as CNP (little trick here)
        # rowNbCNP = testCNP.shape[0]
        # testCP = dfCP.iloc[split:split + rowNbCNP]
        # Xtest = pd.concat([testCP,testCNP])

        Ytest = Xtest['res']
        Xtest = Xtest.drop('res',axis =1)
        return Xtrain,Ytrain,Xtest,Ytest

#I just run python datasetCreation.py and test here
# dc = datasetCreation(skip_int = 0,reload = True)
# df = dc.datasetCreation()
# dfCP = df[df['res'] == 1]
# dfCNP = df[df['res'] == 0]
# X = df[['microssacade','res']]
# X2 = df[['interPupil','res']]
# X3 = df[['blink','res']]
# kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
# kmeans2 = KMeans(n_clusters=4, random_state=0).fit(X2)
# kmeans3 = KMeans(n_clusters=4, random_state=0).fit(X3)
# df['microssacade'] = kmeans.predict(X)
# df['interPupil'] = kmeans2.predict(X2)

# dfCP = shuffle(dfCP)
# dfCNP = shuffle(dfCNP)
#less CNP so 90% used 10% testing
# split = round((dfCNP.shape[0])*.7)
# trainCP,trainCNP = dfCP.iloc[:split],dfCNP.iloc[:split]
# Xtrain = pd.concat([trainCP,trainCNP])
# Ytrain = Xtrain['res']
# Xtrain = Xtrain.drop('res',axis =1)
# testCNP = dfCNP.iloc[split:]
# rowNb = testCNP.shape[0]
# testCP = dfCP.iloc[split:split + rowNb]
# Xtest = pd.concat([testCP,testCNP])
# Ytest = Xtest['res']
# Xtest = Xtest.drop('res',axis =1)
# train,test = df.iloc[:split], df.iloc[split:]
# train_outputs, test_outputs = condition[:split], condition[split:]

# usefull commands
# df.groupby(['blink','res']).size()
# df['word'].value_counts()
# sort= df.sort_values(['blink','res'],ascending = True)
# with pd.option_context('display.max_rows', None, 'display.max_columns', None): print(sort)
