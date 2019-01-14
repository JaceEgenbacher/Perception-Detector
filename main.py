from datasetCreation import datasetCreation
import pandas as pd
import time
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers
from sklearn.metrics import classification_report,roc_auc_score,confusion_matrix,precision_score
import numpy as np
import matplotlib.pyplot as plt


np.random.seed(7)
skip_int = 0

#use to concatenate all then save into csv file
def save(skip_int) :
    dataC = datasetCreation(skip_int = skip_int,reload = True)
    df = dataC.datasetCreation(20)
    dataC.saveDataframe(df)
#load an existing csv file
def load(skip_int) :
    dataC = datasetCreation(skip_int = skip_int,reload = False)
    df = dataC.loadDataframe()
    Xtrain,Ytrain,Xtest,Ytest = dataC.produceSets(df)
    return Xtrain,Ytrain,Xtest,Ytest

# save(skip_int) #you can comment this line to save time once you have saved it once to work on only the ML model
# #load the model
# Xtrain,Ytrain,Xtest,Ytest  = load(skip_int)

def testBestMean():
    # splitList = [5,10,20,30,40,50,60,70,80,90,100,500]
    splitList = [40,50,60,70,80]
    nbIter = 20
    for splitNumber in splitList :
        precisionL = []
        fpL = []
        tpL = []
        print('SPLIT EVERY',splitNumber)
        dataC = datasetCreation(skip_int = skip_int,reload = True)
        df = dataC.datasetCreation(splitNumber)
        Xtrain,Ytrain,Xtest,Ytest = dataC.produceSets(df)
        for i in range(nbIter):
            precision,fp,tp = results(Xtrain,Ytrain,Xtest,Ytest,20)
            precisionL.append(precision)
            fpL.append(fp)
            tpL.append(tp)
        print('after {0:d} times for a split of {1:d} :\n precision = {2:f}\n fp = {3:f}\n tp = {4:f}'.format(nbIter,splitNumber,sum(precisionL)/len(precisionL),sum(fpL)/len(fpL),sum(tpL)/len(tpL)))
def testEpoch():
    epochList = [20,30,40,50]
    dataC = datasetCreation(skip_int = skip_int,reload = True)
    df = dataC.datasetCreation(50)
    Xtrain,Ytrain,Xtest,Ytest = dataC.produceSets(df)
    for epochNb in epochList :
        nbIter = 10
        precisionL = []
        fpL = []
        tpL = []
        for i in range(nbIter):
            precision,fp,tp = results(Xtrain,Ytrain,Xtest,Ytest,epochNb)
            precisionL.append(precision)
            fpL.append(fp)
            tpL.append(tp)
        print('after {0:d} epochs, precision = {1:f}\n fp = {2:f}\n tp = {3:f}'.format(epochNb,sum(precisionL)/len(precisionL),sum(fpL)/len(fpL),sum(tpL)/len(tpL)))

def getProba():
    dataC = datasetCreation(skip_int = skip_int,reload = True)
    df = dataC.datasetCreation(50)
    Xtrain,Ytrain,Xtest,Ytest = dataC.produceSets(df)
    precision,fp,tp,prediction_prob = results(Xtrain,Ytrain,Xtest,Ytest,20)
    return Ytest,prediction_prob

def results(Xtrain,Ytrain,Xtest,Ytest,epoch):
    model = Sequential()
    model.add(Dense(512, input_dim=Xtrain.shape[1], activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    sgd = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
    model.compile(loss='binary_crossentropy', optimizer=sgd , metrics=['accuracy'])
    model.fit(Xtrain, Ytrain,verbose=0, epochs=epoch)
    predict  = model.predict(Xtest, batch_size=None, verbose=0, steps=None)

    prediction_prob = model.predict(Xtest)
    prediction = np.array([1 if elt > 0.5 else 0 for elt in prediction_prob])
    cm = confusion_matrix(Ytest, prediction, labels=[0, 1])
    report = classification_report(Ytest, prediction)
    precision = precision_score(Ytest, prediction, pos_label=1)
    fp = cm[0][1]
    tp = cm[1][1]
    # print(report)
    # print(cm)

    # pos_prob = prediction_prob[:]
    # thresholds = np.arange(0.0, 1.2, 0.1)
    # true_pos, false_pos = [0]*len(thresholds), [0]*len(thresholds)
    # for pred, y in zip(pos_prob, Ytest):
    #     for i, threshold in enumerate(thresholds):
    #         if pred >= threshold:
    #             # if truth and prediction are both 1
    #             if y == 1:
    #                 true_pos[i] +=1
    #             else:
    #                 # if truth is 0 while prediction is 1
    #                 false_pos[i] += 1
    #         else: break
    #
    # true_pos_rate = [tp/Ytest.value_counts()[0] for tp in true_pos]
    # false_pos_rate = [fp/Ytest.value_counts()[1] for fp in false_pos]
    # print('auc_score :', roc_auc_score(Ytest, pos_prob))

    # plt.figure()
    # lw = 2
    # plt.plot(false_pos_rate, true_pos_rate, color='darkorange',lw=lw)
    # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    # plt.xlim([0.0, 1.0])
    # plt.ylim([0.0, 1.05])
    # plt.xlabel('False Positive Rate')
    # plt.ylabel('True Positive Rate')
    # plt.title('Receiver Operating Characteristic')
    # plt.legend(loc="lower right")
    # plt.show()

    return precision,fp,tp,prediction_prob

def main():
    Ytest,prediction_prob = getProba()
    thresholds = np.arange(0.45, 0.6, 0.01)
    true_pos, false_pos = [0]*len(thresholds), [0]*len(thresholds)
    for pred, y in zip(prediction_prob, Ytest):
        for i, threshold in enumerate(thresholds):
            if pred >= threshold:
                # if truth and prediction are both 1
                if y == 1:
                    true_pos[i] +=1
                else:
                    # if truth is 0 while prediction is 1
                    false_pos[i] += 1
            else: break
        if true_pos[i] > 9*false_pos[i] : break


    prediction = np.array([1 if elt > threshold else 0 for elt in prediction_prob])
    cm = confusion_matrix(Ytest, prediction, labels=[0, 1])
    print(cm)
    precision = precision_score(Ytest, prediction, pos_label=1)
    fp = cm[0][1]
    tp = cm[1][1]
    print('precision : {0:f} \n true positives : {1:d} \n false positive : {2:d}'.format(precision,tp,fp))

main()
