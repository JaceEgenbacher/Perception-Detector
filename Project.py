#Used https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import optimizers
import numpy
# fix random seed for reproducibility
numpy.random.seed(7)
# load data

skip_int = 5800 # Used to get through the data quickly
print("Loading Data starting at row", (skip_int + 1))

quadrant = numpy.loadtxt("quadrant.csv", delimiter=",", skiprows = skip_int)
opacity = numpy.loadtxt("opacity.csv", delimiter=",", skiprows = skip_int)
rawGazeLX= numpy.loadtxt("rawGazeLX.csv", delimiter=",", skiprows = skip_int)
rawGazeLY = numpy.loadtxt("rawGazeLY.csv", delimiter=",", skiprows = skip_int)
rawGazeRX = numpy.loadtxt("rawGazeRX.csv", delimiter=",", skiprows = skip_int)
rawGazeRY = numpy.loadtxt("rawGazeRY.csv", delimiter=",", skiprows = skip_int)
rawBlinkLogicalL = numpy.loadtxt("rawBlinkLogicalL.csv", delimiter=",", skiprows = skip_int)
rawBlinkLogicalR = numpy.loadtxt("rawBlinkLogicalR.csv", delimiter=",", skiprows = skip_int)
microsaccadeBinoc = numpy.loadtxt("microsaccadeBinoc.csv", delimiter=",", skiprows = skip_int)
interpPupilL = numpy.loadtxt("interpPupilL.csv", delimiter=",", skiprows = skip_int)
interpPupilR = numpy.loadtxt("interpPupilR.csv", delimiter=",", skiprows = skip_int)
condition = numpy.loadtxt("condition.csv", delimiter=",", skiprows = skip_int)

print("Data Loaded")

dataset = numpy.column_stack((quadrant, opacity, rawGazeLX, rawGazeLY, rawGazeRX, rawGazeRY, rawBlinkLogicalL, rawBlinkLogicalR, microsaccadeBinoc, interpPupilL, interpPupilR))

split = round((dataset.shape[0])*.9) #Used to split 90% for training
train,test = dataset[:split,:], dataset[split:,:]
train_outputs, test_outputs = condition[:split], condition[split:]
print("Data split")


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
