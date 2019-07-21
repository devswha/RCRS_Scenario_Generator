from myModule import *
from sklearn.metrics import mean_squared_error
from keras.utils.generic_utils import CustomObjectScope
from keras.models import load_model
from math import sqrt
import numpy as np
import cv2
import tensorflow as tf
import keras
from keras.optimizers import Adam

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def lr_schedule(epoch):
    lr = 1e-3
    if epoch >= 15:
        lr = 1e-4
    print('Learning rate: ', lr)
    return lr


mapName = ""
GRID_ROW = ""
GRID_COL = ""

# Check data directory
if not len(sys.argv) is 4:
    print("Usage : python test_model.py [MapName] [GridRow] [GridCol] [TestStart] [TestEnd]")
    exit(1)
else:
    mapName = sys.argv[1]
    GRID_ROW = int(sys.argv[2])
    GRID_COL = int(sys.argv[3])
    if not os.path.exists("%s/%s" % (TEST_GENERATED_IMAGE_DIR, mapName)):
        print("%s is not exist!" % mapName)
        exit(1)

now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print('[INFO][%s] start %s to predict' % (now, mapName))

grid = "%dx%d" % (GRID_ROW, GRID_COL)
npyDir = "%s/video/test/%s/%s" % (DATASET_DIR, mapName, grid)

if not os.path.exists("%s/video/%s/%s" % (RESULTS_DIR, mapName, grid)):
    os.makedirs("%s/video/%s/%s" % (RESULTS_DIR, mapName, grid))



class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=8, dim=(192,256,3), n_channels=3,
                 n_classes=16, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, self.n_classes), dtype=object)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = np.load(ID + '.npy')
            # Store class
            y[i] = self.labels[ID]

        return X, y

#####################################################
# Test DNN model
#####################################################

WIDTH = 256
HEIGHT = 192
num_classes = GRID_COL * GRID_ROW
seq = 9

# Load test data
testImage = []
partition = {}
partition['test'] = []


labels = {}
for dataSetNum in range(1, 1900+1):
    partition['test'].append('%s/test_data_video-%d' % (npyDir, dataSetNum))

testLabel = []
for dataSetNum in range(1, 1900+1):
    testLabelData = np.load('%s/test_data_label-%d.npy' % (npyDir, dataSetNum))
    labels['%s/test_data_video-%d' % (npyDir, dataSetNum)] = np.array(testLabelData, dtype=object)
    testLabel.extend(testLabelData[:])

testLabel = np.array(testLabel)



# Parameters
params = {'dim': (seq, HEIGHT, WIDTH),
          'batch_size': 19,
          'n_classes': num_classes,
          'n_channels': 3,
          'shuffle': True}

# Generators
testing_generator = DataGenerator(partition['test'], labels, **params)
'''
from newModel.myNewModel import InceptionResNet_v2_LSTM
from newModel.myNewModel import ResNet_50_LSTM
from newModel.myNewModel import ResNet_101_LSTM
from newModel.myNewModel import ResNet_152_LSTM
from newModel.myNewModel import Inception_v3_LSTM
from newModel.myNewModel import MobileNet_LSTM
from newModel.myNewModel import Xception_LSTM
from newModel.myNewModel import DenseNet_121_LSTM
from newModel.myNewModel import DenseNet_169_LSTM
from newModel.myNewModel import DenseNet_201_LSTM
from newModel.myNewModel import DenseNet_264_LSTM
from newModel.myNewModel import ResNeXt_50_LSTM
from newModel.myNewModel import ResNeXt_101_LSTM
'''
modelName = 'DenseNet_121_CBAM_LSTM_AUG'
with CustomObjectScope({'relu6': keras.layers.advanced_activations.ReLU(6.),'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
    model = keras.models.load_model("%s_saved_models_video/%s" % (grid, modelName), compile=False)

#model.compile(optimizer=Adam(lr=lr_schedule(0)), loss='mean_squared_error', metrics=['mse'])
model.summary()

#scores = model.evaluate_generator(testing_generator, steps=100)
#print("%s: %f" %(model.metrics_names[1], scores[1]))

# model evaluation


predictions = model.predict_generator(testing_generator, steps=100)
print(predictions.shape)
print(testLabel.shape)

index = 0
totalRMS = 0
rmsList = []
f = open("%s/video/%s/%s/RMSE_%s.txt" % (RESULTS_DIR, mapName, grid, modelName), 'w')
for testSetNum in range(0, 1900):
    rms = sqrt(mean_squared_error(predictions[testSetNum], testLabel[testSetNum]))
    f.write(str(rms) + '\n')
    rmsList.append(rms)
    totalRMS = totalRMS + rms

totalRMS = totalRMS / 1900
maxRMSList = []
for i in range(0,5):
    maxRMS = max(rmsList)
    maxRMSList.append(maxRMS)
    rmsList.remove(maxRMS)

avgMaxRMS = 0
for i in maxRMSList:
    avgMaxRMS = avgMaxRMS + i


print("ModelName: " + modelName)
print(npyDir)
print("top5RMSE: " + str(avgMaxRMS/5))
print("avgRMSE: " + str(totalRMS))
f.close()
