from __future__ import absolute_import, division, print_function
from myModule import *
from tensorflow import reset_default_graph
from matplotlib import pyplot

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

from newModel.myNewModel import C3D_CLSTM_C2D

from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.utils.generic_utils import CustomObjectScope
import numpy as np
import tensorflow as tf
import keras
import inspect
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
mapName = ""
GRID_ROW = ""
GRID_COL = ""

def lr_schedule(epoch):
    lr = 1e-3
    if epoch >= 15:
        lr = 1e-4
    print('Learning rate: ', lr)
    return lr

if not len(sys.argv) is 4:
    print("Usage : python train_model.py [Map name] [Grid row] [Grid col]")
    exit(1)
else:
    mapName = sys.argv[1]
    GRID_ROW = int(sys.argv[2])
    GRID_COL = int(sys.argv[3])

# Load path/class_id video file:
grid = "%dx%d" % (GRID_ROW, GRID_COL)
npyDir = "%s/video/train/%s/%s" % (DATASET_DIR, mapName, grid)

if not os.path.exists("%s/%s/%s" % (MODELS_DIR, mapName, grid)):
    os.makedirs("%s/%s/%s" % (MODELS_DIR, mapName, grid))


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
# Train DNN model
#####################################################
# 70

batch_size = 32
nb_epoch = 6
num_classes = GRID_ROW * GRID_COL
WIDTH = 256
HEIGHT = 192
weight_decay = 0.00005
seq = 9

# Load train data
trainImage = []
partition = {}
partition['train'] = []
partition['validation'] = []
labels = {}
for dataSetNum in range(1, 17000+1):
    partition['train'].append('%s/train_data_video-%d' % (npyDir, dataSetNum))
for dataSetNum in range(17000+1, 19000+1):
    partition['validation'].append('%s/train_data_video-%d' % (npyDir, dataSetNum))

trainLabel = []
for dataSetNum in range(1, 19000+1):
    trainLabelData = np.load('%s/train_data_label-%d.npy' % (npyDir, dataSetNum))
    labels['%s/train_data_video-%d' % (npyDir, dataSetNum)] = np.array(trainLabelData, dtype=object)

k = np.load(partition['train'][0] + '.npy')
print(k.shape)

# Parameters
params = {'dim': (seq, HEIGHT, WIDTH),
          'batch_size': 4,
          'n_classes': num_classes,
          'n_channels': 3,
          'shuffle': True}

# Generators
training_generator = DataGenerator(partition['train'], labels, **params)
validation_generator = DataGenerator(partition['validation'], labels, **params)

model = C3D_CLSTM_C2D(HEIGHT, WIDTH, num_classes, seq)
modelName = 'C3D_CLSTM_C2D'
#model = basic_LSTM(HEIGHT, WIDTH, num_classes, seq)
#model = C3D_CLSTM_C2D(HEIGHT, WIDTH, num_classes, seq, weight_decay)


save_dir = os.path.join(os.getcwd(), '%s_saved_models_video' % grid)
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, modelName)


if not os.path.isdir("%s/video/%s/%s" % (MODELS_DIR, mapName, grid)):
    os.makedirs("%s/video/%s/%s" % (MODELS_DIR, mapName, grid))


model.compile(optimizer=Adam(lr=lr_schedule(0)), loss='mean_squared_error', metrics=['mse'])
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_mean_squared_error',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)
callbacks = [checkpoint, lr_reducer, lr_scheduler]
early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0,
                                               mode='auto', baseline=None, restore_best_weights=True)
history = model.fit_generator(generator=training_generator, validation_data=validation_generator,
                              use_multiprocessing=True, workers=6, epochs=nb_epoch, callbacks=callbacks)
model.save("%s/video/%s/%s/%s.h5" % (MODELS_DIR, mapName, grid, modelName))

#export HDF5_USE_FILE_LOCKING=FALSE
y_vloss = history.history['val_loss']
y_loss = history.history['loss']

x_len = np.arange(len(y_loss))
plt.plot(x_len, y_vloss, marker='.', c='red', label="Validation-set Loss")
plt.plot(x_len, y_loss, marker='.', c='blue', label="Train-set Loss")

plt.legend(loc='upper right')
plt.grid()
plt.xlabel('epoch')
plt.ylabel('loss')
plt.savefig('%s_%s.png' % (modelName, grid))
