from datasetCreation import datasetCreation
import pandas as pd
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers
import numpy

numpy.random.seed(7)
skip_int = 0

#use to concatenate all then save into csv file
def save(skip_int) :
    dataC = datasetCreation(skip_int = skip_int,reload = True)
    df = dataC.datasetCreation()
    dataC.saveDataframe(df)
#load an existing csv file
def load(skip_int) :
    dataC = datasetCreation(skip_int = skip_int,reload = False)
    df = dataC.loadDataframe()
    Xtrain,Ytrain,Xtest,Ytest = dataC.produceSets(df)
    return Xtrain,Ytrain,Xtest,Ytest


save(skip_int) #you can comment this line to save time once you have saved it once
Xtrain,Ytrain,Xtest,Ytest  = load(skip_int)


# create model
model = Sequential()
model.add(Dense(512, input_dim=Xtrain.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.summary()

sgd = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=sgd , metrics=['accuracy'])

model.fit(Xtrain, Ytrain, epochs=20)

# scores = model.evaluate(Xtest, Ytest)
# print(model.metrics_names)
# print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
predict  = model.predict(Xtest, batch_size=None, verbose=0, steps=None)
dfPredict  = pd.DataFrame(predict,columns =['pred'])
results = pd.concat([dfPredict,Ytest.reset_index()],axis=1)

results['TP'] = (results.pred > 0.5) & (results.res == 1)
results['FP'] = (results.pred > 0.5) & (results.res == 0)
print('False positive',results[results['FP'] == True])
#value_counts() [number of False, number of True]
TP = results['TP'].value_counts()[True]
#had an error when no FP so had to do a condition
if True in results['FP'].value_counts().keys().tolist() :
    FP = results['FP'].value_counts()[True]
else :
    FP = 0
precision = TP/(TP+FP)*100
totalNbClassfied = TP + FP
print('precision : ', precision)
print('{0:d} classfied as TP ({1:d} TP and {2:d} FP) out of {3:d} samples ({4:d} CP and {5:d} CNP)'.format(totalNbClassfied,TP,FP,Ytest.size,Ytest.value_counts()[1],Ytest.value_counts()[0]))
# ypred = model.predict_classes(test)
