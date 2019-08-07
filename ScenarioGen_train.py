from socket import *
from select import *
from myModule import *

# Set variables
HOST = "localhost"
PORT = 9998
buf = 10
addressInfo = ""
mapName = ""
INITIAL_TIME = 0

# Set map name
if not len(sys.argv) is 2:
    print("Usage : python ScenarioGen_train.py [Map name]")
    exit(1)
else:
    mapName = sys.argv[1]
    if not os.path.exists("%s/%s" % (BASE_MAP_DIR, mapName)):
        print("%s/%s is not exist!" % (BASE_MAP_DIR, mapName))
        exit(1)


# Load base map.gml file
# Parsing road and building location
now = datetime.datetime.now().strftime("%d %H:%M:%S")
print("[INFO][%s] Start to generate %s #%d ~ #%d maps" % (now, mapName, TRAIN_START_MAP_NUM, TRAIN_END_MAP_NUM))
roadList = []
buildingList = []
baseMapFile = open('%s/%s/map/map.gml' % (BASE_MAP_DIR, mapName), 'r')
for paragraph in baseMapFile:
    lines = paragraph.splitlines()
    for eachLine in lines:
        if eachLine.find('rcr:road gml:id') > 0:
            roadList.append(eachLine.split('"')[1])
        elif eachLine.find('rcr:building gml:id') > 0:
            buildingList.append(eachLine.split('"')[1])
        else:
            pass
RnBList = roadList + buildingList
baseMapFile.close()


print("[INFO] Base map's road and building list is parsed")
print("[INFO] Number of roads : %d" % len(roadList))
print("[INFO] Number of buildings : %d" % len(buildingList))
print("[INFO] Number of civilians : %d" % NUM_OF_CIVILIANS)
print("[INFO] Number of fires : %d" % NUM_OF_FIRES)


# Copy base map.gml file
# Set civilians locations randomly within map's buildings and roads
baseScenarioFile = open('%s/%s/map/scenario.xml' % (BASE_MAP_DIR, mapName), 'r')
for mapNum in range(TRAIN_START_MAP_NUM, TRAIN_END_MAP_NUM+1):
    # Create each maps directory
    mapPath = "%s/raw/train/generated_map/%s/%s_%s/map/" % (DATASET_DIR, mapName, mapName, mapNum)
    configPath = "%s/raw/train/generated_map/%s/%s_%s/config" % (DATASET_DIR, mapName, mapName, mapNum)
    if not os.path.exists(mapPath):
        os.makedirs(mapPath)

    # Copy base map.gml file to each maps directory
    shutil.copy("%s/%s/map/map.gml" % (BASE_MAP_DIR, mapName), mapPath)
    shutil.copytree("%s/%s/config" % (BASE_MAP_DIR, mapName), configPath)

    # Modify the configuration file's random seed
    for line in fileinput.input("%s/common.cfg" % configPath, inplace=True):
        if "random.seed:" in line:
            random.seed(datetime.datetime.now())
            line = "random.seed: %d\n" % random.randrange(1, 100)
        sys.stdout.write(line)

    # Create each scenarios
    scenarioFile = open('%s/scenario.xml' % mapPath, 'w+')
    for paragraph in baseScenarioFile:
        lines = paragraph.splitlines()
        for eachLine in lines:
            if eachLine == "</scenario:scenario>":
                random.seed(datetime.datetime.now())
                for numOfCivilians in range(0, NUM_OF_CIVILIANS):
                    scenarioFile.write('  <scenario:civilian scenario:location="%s"/>\n' % random.choice(buildingList))
                random.seed(datetime.datetime.now())
                for numOfFires in range(0, NUM_OF_FIRES):
                    scenarioFile.write('  <scenario:fire scenario:location="%s"/>\n' % random.choice(buildingList))
                scenarioFile.write("</scenario:scenario>")
            else:
                scenarioFile.write(eachLine+'\n')
                pass
    scenarioFile.close()
    baseScenarioFile.seek(0, os.SEEK_SET)
baseScenarioFile.close()


# Create image
now = datetime.datetime.now().strftime("%d %H:%M:%S")
print("[INFO][%s] Finish to generate %s #%d ~ #%d maps" % (now, mapName, TRAIN_START_MAP_NUM, TRAIN_END_MAP_NUM))
print("[INFO][%s] Start to generate %s #%d ~ #%d map images" % (now, mapName, TRAIN_START_MAP_NUM, TRAIN_END_MAP_NUM))
for imageSetNum in range(TRAIN_START_MAP_NUM, TRAIN_END_MAP_NUM+1):
    # Create each imageSet directory
    imageSetPath = "%s/raw/train/generated_image/%s/%s_%d" % (DATASET_DIR, mapName, mapName, imageSetNum)
    generatedMapPath = "%s/raw/train/generated_map/%s/%s_%d" % (DATASET_DIR, mapName, mapName, imageSetNum)
    if not os.path.exists(imageSetPath):
        os.makedirs(imageSetPath)
        os.makedirs("%s/Image" % imageSetPath)
        os.makedirs("%s/Parse" % imageSetPath)
        os.makedirs("%s/Label" % imageSetPath)
        os.makedirs("%s/Image-CiviliansView" % imageSetPath)

    # Run the Robocup Rescue simulator
    # Map directory : Generated maps
    # Config directory : Configuration file in base map directory
    p = subprocess.Popen(['gnome-terminal', '--disable-factory', '--working-directory=%s/roborescue-v1.2/boot/'
                          % SIMULATOR_DIR, '-e', "./start-comprun.sh -m %s/map -c %s/config"
                          % (generatedMapPath, generatedMapPath)], preexec_fn=os.setpgrp)

    # Run the server
    # Saved simulator's log file each time step
    serverSocket = socket(AF_INET, SOCK_STREAM)
    serverSocket.bind((HOST, PORT))
    serverSocket.listen(10)
    connectionList = [serverSocket]
    now = datetime.datetime.now().strftime("%d %H:%M:%S")
    print('[INFO][%s] Run %s #%d map on simulator' % (now, mapName, imageSetNum))
    while connectionList:
        try:
            read_socket, write_socket, error_socket = select(connectionList, [], [], 2)
            for sock in read_socket:
                # New connect
                if sock == serverSocket:
                    clientSocket, addressInfo = serverSocket.accept()
                    connectionList.append(clientSocket)
                # Read Data
                else:
                    packet = sock.recv(buf)
                    if packet:
                        INITIAL_TIME = packet.decode(encoding='UTF-8')
                        if (int(INITIAL_TIME) % 10) == 0:
                            now = datetime.datetime.now().strftime("%d %H:%M:%S")
                            print('[INFO][%s] Time_%d log is saved.' % (now, int(INITIAL_TIME)))
                    else:
                        if (int(INITIAL_TIME) % 10) == 0:
                            now = datetime.datetime.now().strftime("%d %H:%M:%S")
                            print('[INFO][%s] Time_%d log is saved.' % (now, int(INITIAL_TIME) + 1))
                        connectionList.remove(sock)
                        sock.close()
                        connectionList = 0
                        break
        except KeyboardInterrupt:
            serverSocket.close()

    # Copy simulator's log file to DataSet
    now = datetime.datetime.now().strftime("%d %H:%M:%S")
    print('[INFO][%s] Copy simulation information to data set' % now)

    # Copy log files
    # shutil.copytree("%s/roborescue-v1.2/boot/logs" % SIMULATOR_DIR, "%s/Parse/log" % imageSetPath)

    # Copy map and civilians information
    shutil.copy("%s/mapInfo.txt" % SIMULATOR_DIR, "%s/Parse" % imageSetPath)
    shutil.copy("%s/civilianHP.txt" % SIMULATOR_DIR, "%s/Parse" % imageSetPath)
    shutil.copy("%s/civilianLoc.txt" % SIMULATOR_DIR, "%s/Parse" % imageSetPath)

    # Kill simulator and close server
    os.killpg(p.pid, signal.SIGINT)
    serverSocket.close()

    # Revise simulation's configuration map's name
    now = datetime.datetime.now().strftime("%d %H:%M:%S")
    print('[INFO][%s] Copy simulation images to data set' % now)
    for line in fileinput.input("%s/config.txt" % SIMULATOR_DIR, inplace=True):
        print(line.rstrip().replace('NAME_OF_MAPS', 'NAME_OF_MAPS %s_%d' % (mapName, imageSetNum)))
    for line in fileinput.input("%s/config.txt" % SIMULATOR_DIR, inplace=True):
        print(line.rstrip().replace('GRID_DRAW true', 'GRID_DRAW false'))
    for line in fileinput.input("%s/config.txt" % SIMULATOR_DIR, inplace=True):
        print(line.rstrip().replace('CIVILIAN_DRAW true', 'CIVILIAN_DRAW false'))

    # Crop and resize image
    os.chdir("%s/roborescue-v1.2/boot/" % SIMULATOR_DIR)
    os.system("./logextract.sh ./logs/rescue.log %s/Image >/dev/null 2>&1" % imageSetPath)
    os.chdir(os.path.abspath('./'))

    # Revise simulation's configuration map's name
    for line in fileinput.input("%s/config.txt" % SIMULATOR_DIR, inplace=True):
        print(line.rstrip().replace('NAME_OF_MAPS', 'NAME_OF_MAPS %s_%d' % (mapName, imageSetNum)))
    for line in fileinput.input("%s/config.txt" % SIMULATOR_DIR, inplace=True):
        print(line.rstrip().replace('GRID_DRAW false', 'GRID_DRAW true'))
    for line in fileinput.input("%s/config.txt" % SIMULATOR_DIR, inplace=True):
        print(line.rstrip().replace('CIVILIAN_DRAW false', 'CIVILIAN_DRAW true'))

    # Crop and resize image
    os.chdir("%s/roborescue-v1.2/boot/" % SIMULATOR_DIR)
    os.system("./logextract.sh ./logs/rescue.log %s/Image-CiviliansView >/dev/null 2>&1" % imageSetPath)
    os.chdir(os.path.abspath('./'))

now = datetime.datetime.now().strftime("%d %H:%M:%S")
print("[INFO][%s] Finish to generate %s #%d ~ #%d map images" % (now, mapName, TRAIN_START_MAP_NUM, TRAIN_END_MAP_NUM))
sys.exit()



