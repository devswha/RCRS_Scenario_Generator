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

# Load path/class_id image file:
grid = "%dx%d" % (GRID_ROW, GRID_COL)

npyDir = "%s/image/test/%s/%s" % (DATASET_DIR, mapName, grid)
if not os.path.exists(npyDir):
    os.makedirs(npyDir)


#######################################
# Create numpy testing data
#######################################
WIDTH = 256
HEIGHT = 192
pathAndLabelListFile = open("%s/pathAndLabelListFile.txt" % npyDir, "w+")
npyNumIndex = 1
imageNumIndex = 0
npyTrainImageData = []
npyTrainLabelData = []

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
    randomChoiceIndex = pathList.index(random.choice(pathList))

    # Save image to npy
    randomChoiceImage = pathList[randomChoiceIndex]
    screen = cv2.imread(randomChoiceImage, cv2.IMREAD_COLOR)
    screen = cv2.resize(screen, (WIDTH, HEIGHT))
    screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
    npyTrainImageData.append(screen)

    # Save label to npy
    randomChoiceLabel = labelList[randomChoiceIndex]
    npyTrainLabelData.append(randomChoiceLabel)

    # Save npy data
    imageNumIndex = imageNumIndex + 1
    if imageNumIndex == 100:
        # image
        testImageDataFile = "%s/test_data_image-%d.npy" % (npyDir, npyNumIndex)
        np.save(testImageDataFile, npyTrainImageData)
        print(testImageDataFile)
        npyTrainImageData = []

        # label
        testLabelDataFile = "%s/test_data_label-%d.npy" % (npyDir, npyNumIndex)
        np.save(testLabelDataFile, npyTrainLabelData)
        print(testLabelDataFile)
        npyTrainLabelData = []

        imageNumIndex = 0
        npyNumIndex = npyNumIndex + 1

