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
    print("Usage : python train_data_generator.py [Map name]")
    exit(1)
else:
    mapName = sys.argv[1]
    if not os.path.exists("%s/%s" % (BASE_MAP_DIR, mapName)):
        print("%s/%s is not exist!" % (BASE_MAP_DIR, mapName))
        exit(1)


# Load base map.gml file
# Parsing road and building location
now = datetime.datetime.now().strftime("%d %H:%M:%S")
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


# Run the Robocup Rescue simulator
# Map directory : Generated maps
# Config directory : Configuration file in base map directory
p = subprocess.Popen(['gnome-terminal', '--disable-factory', '--working-directory=%s/roborescue-v1.2/boot/'
                      % SIMULATOR_DIR, '-e', "./start-comprun.sh -m %s/%s/map -c %s/%s/config"
                      % (BASE_MAP_DIR, mapName, BASE_MAP_DIR, mapName)], preexec_fn=os.setpgrp)

# Run the server
# Saved simulator's log file each time step
serverSocket = socket(AF_INET, SOCK_STREAM)
serverSocket.bind((HOST, PORT))
serverSocket.listen(10)
connectionList = [serverSocket]
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
                else:
                    connectionList.remove(sock)
                    sock.close()
                    connectionList = 0
                    break
    except KeyboardInterrupt:
        serverSocket.close()

sys.exit()



