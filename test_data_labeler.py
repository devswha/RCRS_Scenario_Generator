from myModule import *

mapName = ""
GRID_COL = 0
GRID_ROW = 0

if not len(sys.argv) is 4:
    print("Usage : python train_data_labeler.py [Map name][Grid row][Grid col]")
    exit(1)
else:
    mapName = sys.argv[1]
    GRID_ROW = int(sys.argv[2])
    GRID_COL = int(sys.argv[3])

# Label matrix
# [ Number of civilians ]
LABEL = [[0]*GRID_COL for i in range(GRID_ROW)]

# Grid matrix
# [ minX, maxX, minY, maxY ]
GRID = [[[0, 0, 0, 0]]*GRID_COL for i in range(GRID_ROW)]
grid = "%dx%d" % (GRID_ROW, GRID_COL)
if not os.path.exists("%s/%s/%s" % (MODELS_DIR, mapName, grid)):
    os.makedirs("%s/%s/%s" % (MODELS_DIR, mapName, grid))

# Log
now = datetime.datetime.now().strftime("%d %H:%M:%S")
print('[INFO][%s] Image label start' % now)
print('[INFO] Data sets root: %s/%s' % (TEST_GENERATED_IMAGE_DIR, mapName))
print('[INFO] Label %s #%d ~ #%d data sets' % (mapName, TEST_START_MAP_NUM, TEST_END_MAP_NUM))

# Filtering and Labeling the images
# Filtering : crop and resize
# Labeling : depend on the time step limit and civilians HP
HPListIndex = 0
for dataSetNum in range(TEST_START_MAP_NUM, TEST_END_MAP_NUM+1):
    dataSetPath = "%s/%s/%s_%d" % (TEST_GENERATED_IMAGE_DIR, mapName, mapName, dataSetNum)

    # Read information files for labeling
    mapInfoFile = open("%s/Parse/mapInfo.txt" % dataSetPath, 'r')
    civilianLocFile = open("%s/Parse/civilianLoc.txt" % dataSetPath, 'r')
    civilianHPFile = open("%s/Parse/civilianHP.txt" % dataSetPath, 'r')

    #
    if os.path.exists("%s/Label/%s" % (dataSetPath, grid)):
        shutil.rmtree("%s/Label/%s" % (dataSetPath, grid))
    os.makedirs("%s/Label/%s" % (dataSetPath, grid))

    # Set map's width and height
    initWidth = 0
    initHeight = 0
    endWidth = mapInfoFile.readline().strip('\n')
    endHeight = mapInfoFile.readline().strip('\n')

    # Assign grid cell's range
    for row in range(0, GRID_ROW):
        for col in range(0, GRID_COL):
            GRID[row][col] = [(col * (int(float(endWidth)) / GRID_COL)),
                              ((col + 1) * (int(float(endWidth)) / GRID_COL)),
                              ((GRID_ROW - (row + 1)) * (int(float(endHeight)) / GRID_ROW)),
                              ((GRID_ROW - row) * (int(float(endHeight)) / GRID_ROW))]

    # Labeling the images
    imageListFile = open("%s/Label/%s/ImageList.txt" % (dataSetPath, grid), "w+")
    while True:
        # Parsing the map data
        # Read time step
        line = civilianLocFile.readline().strip('\n')
        if not line: break
        step = int(line)
        civilianHPFile.readline()
        # Read civilian locations and HP list
        HPList = civilianHPFile.readline().split(' ')
        locList = civilianLocFile.readline().split('>')
        for eachLoc in locList:
            eachLoc = eachLoc.strip(' ')
            eachLoc = eachLoc.strip('<')
            if eachLoc != '' and eachLoc != '\n':
                eachLoc = eachLoc.split(', ')
                x = int(eachLoc[0])
                y = int(eachLoc[1])

                # Check civilian's coordinate within cell
                for row in range(0, GRID_ROW):
                    for col in range(0, GRID_COL):
                        if GRID[row][col][0] <= x <= GRID[row][col][1] and GRID[row][col][2] <= y <= GRID[row][col][3]:
                            if int(HPList[HPListIndex]) <= LIMIT_CIVILIAN_HP:
                                LABEL[row][col] = int(LABEL[row][col]) + 1
            HPListIndex = HPListIndex + 1

        # Copy converted image to data set's label directory
        # Longer than LIMIT_TIME_STEP
        if step > LIMIT_TIME_STEP:
            label = []
            for row in range(0, GRID_ROW):
                for col in range(0, GRID_COL):
                    label.append(int(LABEL[row][col]))
            image = "%s/Image/%s_%d_Time_%d.png" % (dataSetPath, mapName, dataSetNum, step)
            if os.path.isfile("%s" % image):
                imageListFile.write(image)
                for eachLabel in label:
                    imageListFile.write(" " + str(eachLabel))
                imageListFile.write("\n")
            else:
                print("[INFO] %s %s is not exist." % (dataSetPath, image))

        # Initialize label matrix
        LABEL = [[0] * GRID_COL for i in range(GRID_ROW)]
        HPListIndex = 0
        now = datetime.datetime.now().strftime("%d %H:%M:%S")
    imageListFile.close()
    print('[INFO][%s] %s_%d images are labeled' % (now, mapName, dataSetNum))
