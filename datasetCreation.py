import pandas as pd
import numpy as np
import time
from sklearn.utils import shuffle
from sklearn.cluster import KMeans
#Screen dimensions : 1280,768 --> replace incoherent values by the middle of the screen
class datasetCreation :

    def __init__(self,skip_int = 0,reload = False) :
        if reload == True :
            tBeforeL = time.time()
            # self.quadrant = pd.read_csv("quadrant.csv", sep=",",header = None, skiprows = skip_int, names  = ['quadrant'])
            # self.rawGazeLX= pd.read_csv("rawGazeLX.csv", sep=",",header = None, skiprows = skip_int,usecols = [*range(6000,9000)])
            # self.rawGazeLY = pd.read_csv("rawGazeRX.csv", sep=",",header = None, skiprows = skip_int,usecols = [*range(6000,9000)])
            self.rawBlinkLogicalL = pd.read_csv("rawBlinkLogicalL.csv",header = None, sep=",", skiprows = skip_int,usecols = [*range(6000,9000)])
            self.microsaccadeBinoc = pd.read_csv("microsaccadeBinoc.csv",header = None, sep=",", skiprows = skip_int,usecols = [*range(6000,9000)])
            self.interpPupilL = pd.read_csv("interpPupilL.csv", sep=",",header = None, skiprows = skip_int,usecols = [*range(6000,9000)])
            self.condition = pd.read_csv("condition.csv", sep=",",header = None, skiprows = skip_int, names = ['res'])
            tAfterL = time.time()
            print('finished loading in : ', tAfterL - tBeforeL)



    def fillIncohrences(self,df) :
        for index, row in df.iterrows():
            row = row.replace(to_replace = 100000000.0, value = np.NaN)
            mean = row.mean(axis=0,skipna = True)
            row = row.fillna(mean)
            df.at[index] = row
        return df

    def replaceBlink(self,value):
        if value == 0 :
            return 0
        else :
            return 1
    def replaceQuartiles(self,value,q1,median,q4) :
        if value < q1 :
            return 0
        elif value <= median and value > q1:
            return 1
        elif value <= q4 and value > median :
            return 2
        else :
            return 3

    def datasetCreation(self):
        tBeforeC = time.time()
        #self.rawGazeLX = self.fillIncohrences(self.rawGazeLX)
        #self.rawGazeLY = self.fillIncohrences(self.rawGazeLY)
        self.condition[self.condition['res']=='CP'] = 1
        self.condition[self.condition['res']=='CNP'] = 0
        self.rawBlinkLogicalL = self.rawBlinkLogicalL.groupby(np.arange(len(self.rawBlinkLogicalL.columns))//3000, axis=1).mean()*100
        self.microsaccadeBinoc = self.microsaccadeBinoc.groupby(np.arange(len(self.microsaccadeBinoc.columns))//3000, axis=1).mean()*100
        self.interpPupilL = self.interpPupilL.groupby(np.arange(len(self.interpPupilL.columns))//3000, axis=1).mean()
        df = pd.concat([self.rawBlinkLogicalL,self.microsaccadeBinoc,self.interpPupilL,self.condition],axis=1,join = 'inner')
        df.columns = ['blink','microssacade','interPupil','res']
        df['blink'] = df['blink'].apply(self.replaceBlink)
        Xb,Xm,Xi = df[['blink','res']],df[['microssacade','res']],df[['interPupil','res']]
        kmeansb,kmeansm,kmeansi = KMeans(n_clusters=3, random_state=0).fit(Xb),KMeans(n_clusters=4, random_state=0).fit(Xm),KMeans(n_clusters=4, random_state=0).fit(Xi)
        df['blink'] = kmeansb.predict(Xb)
        df['microssacade'] = kmeansm.predict(Xm)
        df['interPupil'] = kmeansi.predict(Xi)
        print('kmeansB',kmeansb.cluster_centers_)
        print('kmeansM',kmeansm.cluster_centers_)
        print('kmeansI',kmeansi.cluster_centers_)
        # q1M,q1I = df['microssacade'].quantile(0.25),df['interPupil'].quantile(0.25)
        # medianM,medianI = df['microssacade'].quantile(0.5),df['interPupil'].quantile(0.5)
        # q4M,q4I = df['microssacade'].quantile(0.75),df['interPupil'].quantile(0.75)
        # df['microssacade'] = df['microssacade'].apply(self.replaceQuartiles,args = (q1M,medianM,q4M))
        # df['interPupil'] = df['interPupil'].apply(self.replaceQuartiles,args = (q1I,medianI,q4I))
        tAfterC = time.time()
        print('finished cleaning in : ', tAfterC - tBeforeC)
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
        dfCP = df[df['res'] == 1]
        dfCNP = df[df['res'] == 0]
        # dfCP = shuffle(dfCP)
        # dfCNP = shuffle(dfCNP)
        #less CNP so 90% used 10% testing
        #issue : we want to train on same number of CP than CNP
        split = round((dfCNP.shape[0])*.7)
        trainCP,trainCNP = dfCP.iloc[:split],dfCNP.iloc[:split]
        Xtrain = pd.concat([trainCP,trainCNP])
        Ytrain = Xtrain['res']
        Xtrain = Xtrain.drop('res',axis =1)
        testCNP = dfCNP.iloc[split:]
        #we want as much CP as CNP
        rowNb = testCNP.shape[0]
        testCP = dfCP.iloc[split:split + rowNb]
        Xtest = pd.concat([testCP,testCNP])
        Ytest = Xtest['res']
        Xtest = Xtest.drop('res',axis =1)
        return Xtrain,Ytrain,Xtest,Ytest
# def replaceBlink(value):
#     if value == 0 :
#         return 0
#     else :
#         return 1

# def replaceMicro(column):
    # q1 = column.quantile(0.25)
    # median = column.quantile(0.5)
    # q4 = column.quantile(0.75)
#     column = column.apply(replaceQuartiles,args = (q1,median,q4))

dc = datasetCreation(skip_int = 0,reload = True)
df = dc.datasetCreation()
dfCP = df[df['res'] == 1]
dfCNP = df[df['res'] == 0]
X = df[['microssacade','res']]
X2 = df[['interPupil','res']]
X3 = df[['blink','res']]
kmeans = KMeans(n_clusters=4, random_state=0).fit(X)
kmeans2 = KMeans(n_clusters=4, random_state=0).fit(X2)
kmeans3 = KMeans(n_clusters=4, random_state=0).fit(X3)
df['microssacade'] = kmeans.predict(X)
df['interPupil'] = kmeans2.predict(X2)

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


# df.groupby(['blink','res']).size()
# df['word'].value_counts()
# sort= df.sort_values(['blink','res'],ascending = True)
# with pd.option_context('display.max_rows', None, 'display.max_columns', None): print(sort)
