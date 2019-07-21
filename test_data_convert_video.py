from myModule import *
import numpy as np
import cv2

mapName = ""
GRID_ROW = ""
GRID_COL = ""

if not len(sys.argv) is 4:
    print("Usage : python test_dataset_converter.py [Map name][Grid row][Grid col]")
    exit(1)
else:
    mapName = sys.argv[1]
    GRID_ROW = int(sys.argv[2])
    GRID_COL = int(sys.argv[3])

# Load path/class_id video file:
grid = "%dx%d" % (GRID_ROW, GRID_COL)

npyDir = "%s/video/test/%s/%s" % (DATASET_DIR, mapName, grid)
if not os.path.exists(npyDir):
    os.makedirs(npyDir)


#######################################
# Create numpy testing data
#######################################
WIDTH = 256
HEIGHT = 192
seq_len = SEQUENCE_TIME_STEP
pathAndLabelListFile = open("%s/pathAndLabelListFile.txt" % npyDir, "w+")
npyTrainImageData = []
npyTrainLabelData = []
imageNumIndex = 0
npyNumIndex = 1

for dataSetNum in range(TEST_START_MAP_NUM, TEST_END_MAP_NUM + 1):
    # Select random time step
    pathList = []
    labelList = []
    dataSetPath = "%s/%s/%s_%d" % (TEST_GENERATED_IMAGE_DIR, mapName, mapName, dataSetNum)
    rawImgListFile = open("%s/Label/%s/ImageList.txt" % (dataSetPath, grid), "r")
    for line in rawImgListFile.readlines():
        pathList.append(line.split(' ')[0])
        label = line.split(' ')[1:]
        label[-1] = label[-1].rstrip('\n')
        label = list(map(int, label))
        labelList.append(label)

    # All prediction
    '''
    for i in range(0, 200-seq_len):
        npyTrainSequenceData = []
        # Save video to npy
        for index in range(0+i, seq_len+i):
            screen = cv2.imread(pathList[index], cv2.IMREAD_COLOR)
            screen = cv2.resize(screen, (WIDTH, HEIGHT))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            npyTrainSequenceData.append(screen)
        npyTrainImageData.append(npyTrainSequenceData)

        # Save label to npy
        npyTrainLabelData.append(labelList[seq_len+i-1])
    '''

    for i in range(9, 190, 10):
        # Save video to npy
        npyTrainSequenceData = []
        for index in range(i-9, i):
            screen = cv2.imread(pathList[index], cv2.IMREAD_COLOR)
            screen = cv2.resize(screen, (WIDTH, HEIGHT))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
            npyTrainSequenceData.append(screen)
        npyTrainImageData.append(npyTrainSequenceData)

        # Save label to npy
        npyTrainLabelData.append(labelList[i])
        imageNumIndex = imageNumIndex + 1

        # video
        trainImageDataFile = "%s/test_data_video-%d.npy" % (npyDir, npyNumIndex)
        np.save(trainImageDataFile, npyTrainImageData)
        print(trainImageDataFile)
        npyTrainImageData = []

        # label
        trainLabelDataFile = "%s/test_data_label-%d.npy" % (npyDir, npyNumIndex)
        np.save(trainLabelDataFile, npyTrainLabelData)
        print(trainLabelDataFile)
        npyTrainLabelData = []


        npyNumIndex = npyNumIndex + 1
