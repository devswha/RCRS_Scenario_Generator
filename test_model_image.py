from myModule import *
from sklearn.metrics import mean_squared_error
from keras.utils.generic_utils import CustomObjectScope
from keras.models import load_model
from math import sqrt
import numpy as np
import cv2
import tensorflow as tf
import keras

'''

from newModel.myNewModel import Inception_v3
from newModel.myNewModel import InceptionResNet_v2
from newModel.myNewModel import InceptionResNet_v2_CBAM
from newModel.myNewModel import InceptionResNet_v2_SE
from newModel.myNewModel import InceptionResNet_v2_GCBAM
from newModel.myNewModel import ResNet_50
from newModel.myNewModel import ResNet_50_CBAM
from newModel.myNewModel import ResNet_50_GCBAM
from newModel.myNewModel import ResNet_50_SE
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
'''

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


mapName = ""
GRID_ROW = ""
GRID_COL = ""

# Check data directory
if not len(sys.argv) is 4:
    print("Usage : python test_model.py [MapName] [GridRow] [GridCol]")
    exit(1)
else:
    mapName = sys.argv[1]
    GRID_ROW = int(sys.argv[2])
    GRID_COL = int(sys.argv[3])
    if not os.path.exists("%s/%s" % (TEST_GENERATED_IMAGE_DIR, mapName)):
        print("%s is not exist!" % mapName)
        exit(1)


DATASET_DIR = "./dataset2"
now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print('[INFO][%s] start %s to predict' % (now, mapName))

grid = "%dx%d" % (GRID_ROW, GRID_COL)
npyDir = "%s/image/test/%s/%s" % (DATASET_DIR, mapName, grid)

if not os.path.exists("%s/image/%s/%s" % (RESULTS_DIR, mapName, grid)):
    os.makedirs("%s/image/%s/%s" % (RESULTS_DIR, mapName, grid))

#####################################################
# Test DNN model
#####################################################

# Load test data
testImage = []
for dataSetNum in range(1, int(TEST_END_MAP_NUM/100)+1):
    testImageData = np.load('%s/test_data_image-%d.npy' % (npyDir, dataSetNum))
    testImage.extend(testImageData[:])
test_X = np.array(testImage)

testLabel = []
for dataSetNum in range(1, int(TEST_END_MAP_NUM/100)+1):
    testLabelData = np.load('%s/test_data_label-%d.npy' % (npyDir, dataSetNum))
    testLabel.extend(testLabelData[:])
test_Y = np.array(testLabel)


print("test_X shape: ", test_X.shape)
print("test_Y shape: ", test_Y.shape)


WIDTH = 256
HEIGHT = 192
num_classes = GRID_COL * GRID_ROW
modelName = 'MobileNet'

with CustomObjectScope({'relu6': keras.layers.advanced_activations.ReLU(6.),'DepthwiseConv2D': keras.layers.DepthwiseConv2D}):
    model = keras.models.load_model("%s/image/%s/%s/%s.h5" % (MODELS_DIR, mapName, grid, modelName), compile=False)

predictions = model.predict(test_X)

model.summary()

print(predictions[0])
print(test_Y[0])


index = 0
totalRMS = 0
rmsList = []
f = open("%s/image/%s/%s/RMSE_%s.txt" % (RESULTS_DIR, mapName, grid, modelName), 'w')
for testSetNum in range(TEST_START_MAP_NUM, TEST_END_MAP_NUM + 1):
    rms = sqrt(mean_squared_error(predictions[testSetNum-1], test_Y[testSetNum-1]))
    f.write(str(rms) + '\n')
    rmsList.append(rms)
    totalRMS = totalRMS + rms

totalRMS = totalRMS / TEST_END_MAP_NUM
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
