from __future__ import absolute_import, division, print_function
from myModule import *
from tensorflow import reset_default_graph
from matplotlib import pyplot

'''
from myModel_image import Xception
from myModel_image import VGG19
from myModel_image import WideResNet28_8

from myModel_image import ResNet50
from myModel_image import ResNet101
from myModel_image import ResNet152
from myModel_image import ResNet50V2
from myModel_image import ResNet101V2
from myModel_image import ResNet152V2
from myModel_image import ResNeXt50
from myModel_image import ResNeXt101
from myModel_image import MobileNet
from myModel_image import InceptionResNetV2
from myModel_image import InceptionV3
from myModel_image import DenseNet201
from myModel_image import DenseNet264
from myModel_image import DenseNet121
from myModel_image import WideResNet28_8
from myModel_image import WideResNet40_10
from myModel_image import WideResNet16_8
from myModel_image import WideResNet34_2
from myModel_image import MobileNet_CBAM
from myModel_image import MobileNet_SE
from myModel_image import MobileNet_CUSTOM
from myModel_image import WideResNet28_8
from myModel_image import WideResNet28_8_SE
from myModel_image import WideResNet28_8_CBAM
from myModel_image import WideResNet28_8_CUSTOM
'''


from newModel.myNewModel import Inception_v3
from newModel.myNewModel import InceptionResNet_v2
from newModel.myNewModel import InceptionResNet_v2_CBAM
from newModel.myNewModel import InceptionResNet_v2_SE
from newModel.myNewModel import InceptionResNet_v2_GCBAM
from newModel.myNewModel import ResNeXt_50
from newModel.myNewModel import ResNeXt_101
from newModel.myNewModel import ResNet_50
from newModel.myNewModel import ResNet_101
from newModel.myNewModel import ResNet_152
from newModel.myNewModel import MobileNet
from newModel.myNewModel import MobileNet_CBAM
from newModel.myNewModel import MobileNet_SE
from newModel.myNewModel import MobileNet_GCBAM
from newModel.myNewModel import DenseNet_121
from newModel.myNewModel import DenseNet_169
from newModel.myNewModel import DenseNet_201
from newModel.myNewModel import DenseNet_264
from newModel.myNewModel import Xception
from newModel.myNewModel import Xception_CBAM
from newModel.myNewModel import Xception_GCBAM
from newModel.myNewModel import Xception_SE
from newModel.myNewModel import WideResNet_28_8


from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf
import keras
import inspect
from sklearn.model_selection import KFold

def lr_schedule(epoch):
    """Learning Rate Schedule
    Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
    Called automatically every epoch as part of callbacks during training.
    # Arguments
        epoch (int): The number of epochs
    # Returns
        lr (float32): learning rate
    """
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print('Learning rate: ', lr)
    return lr

mapName = ""
GRID_ROW = ""
GRID_COL = ""

#DATASET_DIR = "./dataset"


if not len(sys.argv) is 4:
    print("Usage : python train_model.py [Map name] [Grid row] [Grid col]")
    exit(1)
else:
    mapName = sys.argv[1]
    GRID_ROW = int(sys.argv[2])
    GRID_COL = int(sys.argv[3])

# Load path/class_id image file:
grid = "%dx%d" % (GRID_ROW, GRID_COL)
npyDir = "%s/image/train/%s/%s" % (DATASET_DIR, mapName, grid)
if not os.path.exists("%s/image/%s/%s" % (MODELS_DIR, mapName, grid)):
    os.makedirs("%s/image/%s/%s" % (MODELS_DIR, mapName, grid))

print(npyDir)

#####################################################
# Train DNN model
#####################################################

batch_size = 8
nb_epoch = 6
num_classes = GRID_ROW * GRID_COL
WIDTH = 256
HEIGHT = 192

# Load train data
trainImage = []
for dataSetNum in range(1, int(TRAIN_END_MAP_NUM/100)+1):
    trainImageData = np.load('%s/train_data_image-%d.npy' % (npyDir, dataSetNum))
    trainImage.extend(trainImageData[:])
train_X = np.array(trainImage)

trainLabel = []
for dataSetNum in range(1, int(TRAIN_END_MAP_NUM/100)+1):
    trainLabelData = np.load('%s/train_data_label-%d.npy' % (npyDir, dataSetNum))
    trainLabel.extend(trainLabelData[:])
train_Y = np.array(trainLabel)

print("train_X shape: ", train_X.shape)
print("train_Y shape: ", train_Y.shape)



x_train, x_valid, y_train, y_valid = train_test_split(train_X, train_Y, test_size=0.33, shuffle= True)

modelName = 'WideResNet_28_8'
model = WideResNet_28_8(HEIGHT, WIDTH, num_classes)
print(modelName)

save_dir = os.path.join(os.getcwd(), '%s_saved_models' % grid)

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
filepath = os.path.join(save_dir, modelName)




skf = KFold(n_splits=5, shuffle=True)
accuracy = []
for train, validation in skf.split(train_X, train_Y):
    model.compile(optimizer=Adam(lr=lr_schedule(0)), loss='mean_squared_error', metrics=['mse'])
    #early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=4, verbose=1, mode='auto', baseline=None, restore_best_weights=True)
    checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True)

    lr_scheduler = LearningRateScheduler(lr_schedule)
    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                               cooldown=0,
                               patience=5,
                               min_lr=0.5e-6)

    callbacks = [checkpoint, lr_reducer, lr_scheduler]
    hist = model.fit(train_X[train], train_Y[train], validation_data=(train_X[validation], train_Y[validation]),
                     shuffle=True, callbacks=callbacks, epochs=nb_epoch, batch_size=batch_size)
    k_accuracy = '%.4f' % (model.evaluate(train_X[validation], train_Y[validation])[1])
    accuracy.append(k_accuracy) 
model.save("%s/image/%s/%s/%s.h5" % (MODELS_DIR, mapName, grid, modelName))

'''
checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True)

lr_scheduler = LearningRateScheduler(lr_schedule)
lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
                           cooldown=0,
                           patience=5,
                           min_lr=0.5e-6)

callbacks = [checkpoint, lr_reducer, lr_scheduler]


model = ResNet_50(HEIGHT, WIDTH, num_classes)
model.compile(optimizer=Adam(lr=lr_schedule(0)), loss='mean_squared_error', metrics=['mse'])
hist = model.fit(train_X, train_Y, validation_split=0.2, shuffle=True, callbacks=callbacks,
                    epochs=nb_epoch, batch_size=batch_size)
model.save("%s/image/%s/%s/%s.h5" % (MODELS_DIR, mapName, grid, modelName))
#print('\nK-fold cross validation Accuracy: {}'.format(accuracy))

import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()
acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
loss_ax.legend(loc='upper left')

acc_ax.plot(hist.history['mean_squared_error'], 'b', label='train acc')
acc_ax.plot(hist.history['val_mean_squared_error'], 'g', label='val acc')
acc_ax.set_ylabel('mean_squared_error')
acc_ax.legend(loc='upper left')

plt.savefig('%s_%s.png' % (modelName, grid))
'''

