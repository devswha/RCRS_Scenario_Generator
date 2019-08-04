
# Robocup Rescue Simulation Scenario Generator
We developed a scenario generator to train the machine running model. In deep learning with image processing, it takes a lot of image data to train the machine learning model. And it takes a lot of work to generate this amount of data by a human. Therefore, we have developed a scenario generator that automatically generates simulated image data. Our scenario generator run as follows: (1) Input the setting of scenario (i.e., the number of civilians and fires, location of the rescue team, etc.) and the number of scenarios to create; and (2) The generator automatically runs the scenario on RCRS; and (3) As the simulation runs, the generator automatically parses the screenshot image data and text data (i.e., the number of injured civilians, rescue team location, etc.)

## 1. Software Pre-Requisites

- Git
- OpenJDK Java 8+

## 2. Download project from GitHub and decompress RCRS

```bash

$ git clone https://github.com/swhaKo/RCRS-deep-learning.git
$ cd RCRS-deep-learning/
$ unzip RCRS.zip
```
## 3. Configuration

In this repository, there is configuration file called "config.txt"
You can modify the number of civilians, the number of initial fire building and the number of data sets.
Also you can modifiy the path where datasets are stored.

### Map data constant
NUM_OF_CIVILIANS: The number of civilians in the simulation map
NUM_OF_FIRES: The number of initial fire buildings
LIMIT_TIME_STEP: The start point of disaster scenario time step to save image dataset.
LIMIT_CIVILIAN_HP: The treshold of civilians' HP point which determine civilian is injured or not

### Train and test data constant
TRAIN_START_MAP_NUM: The map data start number for training
TRAIN_END_MAP_NUM: The map data end number for training
TEST_START_MAP_NUM: The map data start number for testing
TEST_END_MAP_NUM: The map data end number for testing

### Directory data constant
DATASET_DIR: The path of data set directory
TRAIN_GENERATED_MAP_DIR: The path of the source of simulation map data for training
TRAIN_GENERATED_IMAGE_DIR: The path of the screenshot image of simulation map data for training
TEST_GENERATED_MAP_DIR: The path of the source of simulation map data for testing
TEST_GENERATED_IMAGE_DIR: The path of the screenshot image of simulation map data for testing

## 4. Execute
```bash

$ python3 train_data_generator.py [Map Name]
$ python3 test_data_generator.py [Map Name]
```

## 5. Map List

- Kobe
- Joao
