import pandas as pd
import numpy as np
import time
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

    def datasetCreation(self):
        tBeforeC = time.time()
        #self.rawGazeLX = self.fillIncohrences(self.rawGazeLX)
        #self.rawGazeLY = self.fillIncohrences(self.rawGazeLY)
        self.condition[self.condition['res']=='CP'] = 1
        self.condition[self.condition['res']=='CNP'] = 0
        self.rawBlinkLogicalL = self.rawBlinkLogicalL.groupby(np.arange(len(self.rawBlinkLogicalL.columns))//1000, axis=1).mean()
        self.microsaccadeBinoc = self.microsaccadeBinoc.groupby(np.arange(len(self.microsaccadeBinoc.columns))//1000, axis=1).mean()
        self.interpPupilL = self.interpPupilL.groupby(np.arange(len(self.interpPupilL.columns))//1000, axis=1).mean()
        df = pd.concat([
        #self.quadrant,
        #self.rawGazeLX,self.rawGazeLY,
        self.rawBlinkLogicalL,self.microsaccadeBinoc,self.interpPupilL,self.condition],axis=1,join = 'inner')
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

# dc = datasetCreation(skip_int = 5000,reload = True)
# df = dc.datasetCreation()
# print(df)
