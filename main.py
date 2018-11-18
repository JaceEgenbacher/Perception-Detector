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
    return df

t0 = time.time()
save(skip_int) #you can comment this line to save time once you have saved it once
df  = load(skip_int)
condition = df.pop('res')
X = df
split = round((df.shape[0])*.9) #Used to split 90% for training
train,test = df.iloc[:split], df.iloc[split:]
train_outputs, test_outputs = condition[:split], condition[split:]
print("Data split in : ", time.time() - t0 )

# split into input (X) and output (Z) variables
X = train
Z = train_outputs

# create model
model = Sequential()
model.add(Dense(512, input_dim=X.shape[1], activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(512, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.summary()

sgd = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
model.compile(loss='binary_crossentropy', optimizer=sgd , metrics=['accuracy'])

model.fit(X, Z, epochs=100, batch_size=1000)

scores = model.evaluate(test, test_outputs)

print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
